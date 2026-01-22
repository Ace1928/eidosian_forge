from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
class GrammarParser:
    """
    The parser for Python grammar files.
    """

    def __init__(self, bnf_grammar: str):
        self._bnf_grammar = bnf_grammar
        self.generator = tokenize(bnf_grammar, version_info=parse_version_string('3.9'))
        self._gettoken()

    def parse(self) -> Iterator[Tuple[NFAState, NFAState]]:
        while self.type != PythonTokenTypes.ENDMARKER:
            while self.type == PythonTokenTypes.NEWLINE:
                self._gettoken()
            self._current_rule_name = self._expect(PythonTokenTypes.NAME)
            self._expect(PythonTokenTypes.OP, ':')
            a, z = self._parse_rhs()
            self._expect(PythonTokenTypes.NEWLINE)
            yield (a, z)

    def _parse_rhs(self):
        a, z = self._parse_items()
        if self.value != '|':
            return (a, z)
        else:
            aa = NFAState(self._current_rule_name)
            zz = NFAState(self._current_rule_name)
            while True:
                aa.add_arc(a)
                z.add_arc(zz)
                if self.value != '|':
                    break
                self._gettoken()
                a, z = self._parse_items()
            return (aa, zz)

    def _parse_items(self):
        a, b = self._parse_item()
        while self.type in (PythonTokenTypes.NAME, PythonTokenTypes.STRING) or self.value in ('(', '['):
            c, d = self._parse_item()
            b.add_arc(c)
            b = d
        return (a, b)

    def _parse_item(self):
        if self.value == '[':
            self._gettoken()
            a, z = self._parse_rhs()
            self._expect(PythonTokenTypes.OP, ']')
            a.add_arc(z)
            return (a, z)
        else:
            a, z = self._parse_atom()
            value = self.value
            if value not in ('+', '*'):
                return (a, z)
            self._gettoken()
            z.add_arc(a)
            if value == '+':
                return (a, z)
            else:
                return (a, a)

    def _parse_atom(self):
        if self.value == '(':
            self._gettoken()
            a, z = self._parse_rhs()
            self._expect(PythonTokenTypes.OP, ')')
            return (a, z)
        elif self.type in (PythonTokenTypes.NAME, PythonTokenTypes.STRING):
            a = NFAState(self._current_rule_name)
            z = NFAState(self._current_rule_name)
            a.add_arc(z, self.value)
            self._gettoken()
            return (a, z)
        else:
            self._raise_error('expected (...) or NAME or STRING, got %s/%s', self.type, self.value)

    def _expect(self, type_, value=None):
        if self.type != type_:
            self._raise_error('expected %s, got %s [%s]', type_, self.type, self.value)
        if value is not None and self.value != value:
            self._raise_error('expected %s, got %s', value, self.value)
        value = self.value
        self._gettoken()
        return value

    def _gettoken(self):
        tup = next(self.generator)
        self.type, self.value, self.begin, prefix = tup

    def _raise_error(self, msg, *args):
        if args:
            try:
                msg = msg % args
            except:
                msg = ' '.join([msg] + list(map(str, args)))
        line = self._bnf_grammar.splitlines()[self.begin[0] - 1]
        raise SyntaxError(msg, ('<grammar>', self.begin[0], self.begin[1], line))