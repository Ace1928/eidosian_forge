import os
from typing import (
from blib2to3.pgen2 import grammar, token, tokenize
from blib2to3.pgen2.tokenize import GoodTokenInfo
class ParserGenerator:
    filename: Path
    stream: IO[str]
    generator: Iterator[GoodTokenInfo]
    first: Dict[str, Optional[Dict[str, int]]]

    def __init__(self, filename: Path, stream: Optional[IO[str]]=None) -> None:
        close_stream = None
        if stream is None:
            stream = open(filename, encoding='utf-8')
            close_stream = stream.close
        self.filename = filename
        self.stream = stream
        self.generator = tokenize.generate_tokens(stream.readline)
        self.gettoken()
        self.dfas, self.startsymbol = self.parse()
        if close_stream is not None:
            close_stream()
        self.first = {}
        self.addfirstsets()

    def make_grammar(self) -> PgenGrammar:
        c = PgenGrammar()
        names = list(self.dfas.keys())
        names.sort()
        names.remove(self.startsymbol)
        names.insert(0, self.startsymbol)
        for name in names:
            i = 256 + len(c.symbol2number)
            c.symbol2number[name] = i
            c.number2symbol[i] = name
        for name in names:
            dfa = self.dfas[name]
            states = []
            for state in dfa:
                arcs = []
                for label, next in sorted(state.arcs.items()):
                    arcs.append((self.make_label(c, label), dfa.index(next)))
                if state.isfinal:
                    arcs.append((0, dfa.index(state)))
                states.append(arcs)
            c.states.append(states)
            c.dfas[c.symbol2number[name]] = (states, self.make_first(c, name))
        c.start = c.symbol2number[self.startsymbol]
        return c

    def make_first(self, c: PgenGrammar, name: str) -> Dict[int, int]:
        rawfirst = self.first[name]
        assert rawfirst is not None
        first = {}
        for label in sorted(rawfirst):
            ilabel = self.make_label(c, label)
            first[ilabel] = 1
        return first

    def make_label(self, c: PgenGrammar, label: str) -> int:
        ilabel = len(c.labels)
        if label[0].isalpha():
            if label in c.symbol2number:
                if label in c.symbol2label:
                    return c.symbol2label[label]
                else:
                    c.labels.append((c.symbol2number[label], None))
                    c.symbol2label[label] = ilabel
                    return ilabel
            else:
                itoken = getattr(token, label, None)
                assert isinstance(itoken, int), label
                assert itoken in token.tok_name, label
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel
        else:
            assert label[0] in ('"', "'"), label
            value = eval(label)
            if value[0].isalpha():
                if label[0] == '"':
                    keywords = c.soft_keywords
                else:
                    keywords = c.keywords
                if value in keywords:
                    return keywords[value]
                else:
                    c.labels.append((token.NAME, value))
                    keywords[value] = ilabel
                    return ilabel
            else:
                itoken = grammar.opmap[value]
                if itoken in c.tokens:
                    return c.tokens[itoken]
                else:
                    c.labels.append((itoken, None))
                    c.tokens[itoken] = ilabel
                    return ilabel

    def addfirstsets(self) -> None:
        names = list(self.dfas.keys())
        names.sort()
        for name in names:
            if name not in self.first:
                self.calcfirst(name)

    def calcfirst(self, name: str) -> None:
        dfa = self.dfas[name]
        self.first[name] = None
        state = dfa[0]
        totalset: Dict[str, int] = {}
        overlapcheck = {}
        for label in state.arcs:
            if label in self.dfas:
                if label in self.first:
                    fset = self.first[label]
                    if fset is None:
                        raise ValueError('recursion for rule %r' % name)
                else:
                    self.calcfirst(label)
                    fset = self.first[label]
                    assert fset is not None
                totalset.update(fset)
                overlapcheck[label] = fset
            else:
                totalset[label] = 1
                overlapcheck[label] = {label: 1}
        inverse: Dict[str, str] = {}
        for label, itsfirst in overlapcheck.items():
            for symbol in itsfirst:
                if symbol in inverse:
                    raise ValueError('rule %s is ambiguous; %s is in the first sets of %s as well as %s' % (name, symbol, label, inverse[symbol]))
                inverse[symbol] = label
        self.first[name] = totalset

    def parse(self) -> Tuple[Dict[str, List['DFAState']], str]:
        dfas = {}
        startsymbol: Optional[str] = None
        while self.type != token.ENDMARKER:
            while self.type == token.NEWLINE:
                self.gettoken()
            name = self.expect(token.NAME)
            self.expect(token.OP, ':')
            a, z = self.parse_rhs()
            self.expect(token.NEWLINE)
            dfa = self.make_dfa(a, z)
            self.simplify_dfa(dfa)
            dfas[name] = dfa
            if startsymbol is None:
                startsymbol = name
        assert startsymbol is not None
        return (dfas, startsymbol)

    def make_dfa(self, start: 'NFAState', finish: 'NFAState') -> List['DFAState']:
        assert isinstance(start, NFAState)
        assert isinstance(finish, NFAState)

        def closure(state: NFAState) -> Dict[NFAState, int]:
            base: Dict[NFAState, int] = {}
            addclosure(state, base)
            return base

        def addclosure(state: NFAState, base: Dict[NFAState, int]) -> None:
            assert isinstance(state, NFAState)
            if state in base:
                return
            base[state] = 1
            for label, next in state.arcs:
                if label is None:
                    addclosure(next, base)
        states = [DFAState(closure(start), finish)]
        for state in states:
            arcs: Dict[str, Dict[NFAState, int]] = {}
            for nfastate in state.nfaset:
                for label, next in nfastate.arcs:
                    if label is not None:
                        addclosure(next, arcs.setdefault(label, {}))
            for label, nfaset in sorted(arcs.items()):
                for st in states:
                    if st.nfaset == nfaset:
                        break
                else:
                    st = DFAState(nfaset, finish)
                    states.append(st)
                state.addarc(st, label)
        return states

    def dump_nfa(self, name: str, start: 'NFAState', finish: 'NFAState') -> None:
        print('Dump of NFA for', name)
        todo = [start]
        for i, state in enumerate(todo):
            print('  State', i, state is finish and '(final)' or '')
            for label, next in state.arcs:
                if next in todo:
                    j = todo.index(next)
                else:
                    j = len(todo)
                    todo.append(next)
                if label is None:
                    print('    -> %d' % j)
                else:
                    print('    %s -> %d' % (label, j))

    def dump_dfa(self, name: str, dfa: Sequence['DFAState']) -> None:
        print('Dump of DFA for', name)
        for i, state in enumerate(dfa):
            print('  State', i, state.isfinal and '(final)' or '')
            for label, next in sorted(state.arcs.items()):
                print('    %s -> %d' % (label, dfa.index(next)))

    def simplify_dfa(self, dfa: List['DFAState']) -> None:
        changes = True
        while changes:
            changes = False
            for i, state_i in enumerate(dfa):
                for j in range(i + 1, len(dfa)):
                    state_j = dfa[j]
                    if state_i == state_j:
                        del dfa[j]
                        for state in dfa:
                            state.unifystate(state_j, state_i)
                        changes = True
                        break

    def parse_rhs(self) -> Tuple['NFAState', 'NFAState']:
        a, z = self.parse_alt()
        if self.value != '|':
            return (a, z)
        else:
            aa = NFAState()
            zz = NFAState()
            aa.addarc(a)
            z.addarc(zz)
            while self.value == '|':
                self.gettoken()
                a, z = self.parse_alt()
                aa.addarc(a)
                z.addarc(zz)
            return (aa, zz)

    def parse_alt(self) -> Tuple['NFAState', 'NFAState']:
        a, b = self.parse_item()
        while self.value in ('(', '[') or self.type in (token.NAME, token.STRING):
            c, d = self.parse_item()
            b.addarc(c)
            b = d
        return (a, b)

    def parse_item(self) -> Tuple['NFAState', 'NFAState']:
        if self.value == '[':
            self.gettoken()
            a, z = self.parse_rhs()
            self.expect(token.OP, ']')
            a.addarc(z)
            return (a, z)
        else:
            a, z = self.parse_atom()
            value = self.value
            if value not in ('+', '*'):
                return (a, z)
            self.gettoken()
            z.addarc(a)
            if value == '+':
                return (a, z)
            else:
                return (a, a)

    def parse_atom(self) -> Tuple['NFAState', 'NFAState']:
        if self.value == '(':
            self.gettoken()
            a, z = self.parse_rhs()
            self.expect(token.OP, ')')
            return (a, z)
        elif self.type in (token.NAME, token.STRING):
            a = NFAState()
            z = NFAState()
            a.addarc(z, self.value)
            self.gettoken()
            return (a, z)
        else:
            self.raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)
            raise AssertionError

    def expect(self, type: int, value: Optional[Any]=None) -> str:
        if self.type != type or (value is not None and self.value != value):
            self.raise_error('expected %s/%s, got %s/%s', type, value, self.type, self.value)
        value = self.value
        self.gettoken()
        return value

    def gettoken(self) -> None:
        tup = next(self.generator)
        while tup[0] in (tokenize.COMMENT, tokenize.NL):
            tup = next(self.generator)
        self.type, self.value, self.begin, self.end, self.line = tup

    def raise_error(self, msg: str, *args: Any) -> NoReturn:
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = ' '.join([msg] + list(map(str, args)))
        raise SyntaxError(msg, (self.filename, self.end[0], self.end[1], self.line))