from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
class LALR_Analyzer(GrammarAnalyzer):
    lr0_itemsets: Set[LR0ItemSet]
    nonterminal_transitions: List[Tuple[LR0ItemSet, Symbol]]
    lookback: Dict[Tuple[LR0ItemSet, Symbol], Set[Tuple[LR0ItemSet, Rule]]]
    includes: Dict[Tuple[LR0ItemSet, Symbol], Set[Tuple[LR0ItemSet, Symbol]]]
    reads: Dict[Tuple[LR0ItemSet, Symbol], Set[Tuple[LR0ItemSet, Symbol]]]
    directly_reads: Dict[Tuple[LR0ItemSet, Symbol], Set[Symbol]]

    def __init__(self, parser_conf: ParserConf, debug: bool=False, strict: bool=False):
        GrammarAnalyzer.__init__(self, parser_conf, debug, strict)
        self.nonterminal_transitions = []
        self.directly_reads = defaultdict(set)
        self.reads = defaultdict(set)
        self.includes = defaultdict(set)
        self.lookback = defaultdict(set)

    def compute_lr0_states(self) -> None:
        self.lr0_itemsets = set()
        cache: Dict['State', LR0ItemSet] = {}

        def step(state: LR0ItemSet) -> Iterator[LR0ItemSet]:
            _, unsat = classify_bool(state.closure, lambda rp: rp.is_satisfied)
            d = classify(unsat, lambda rp: rp.next)
            for sym, rps in d.items():
                kernel = fzset({rp.advance(sym) for rp in rps})
                new_state = cache.get(kernel, None)
                if new_state is None:
                    closure = set(kernel)
                    for rp in kernel:
                        if not rp.is_satisfied and (not rp.next.is_term):
                            closure |= self.expand_rule(rp.next, self.lr0_rules_by_origin)
                    new_state = LR0ItemSet(kernel, closure)
                    cache[kernel] = new_state
                state.transitions[sym] = new_state
                yield new_state
            self.lr0_itemsets.add(state)
        for _ in bfs(self.lr0_start_states.values(), step):
            pass

    def compute_reads_relations(self):
        for root in self.lr0_start_states.values():
            assert len(root.kernel) == 1
            for rp in root.kernel:
                assert rp.index == 0
                self.directly_reads[root, rp.next] = set([Terminal('$END')])
        for state in self.lr0_itemsets:
            seen = set()
            for rp in state.closure:
                if rp.is_satisfied:
                    continue
                s = rp.next
                if s not in self.lr0_rules_by_origin:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                nt = (state, s)
                self.nonterminal_transitions.append(nt)
                dr = self.directly_reads[nt]
                r = self.reads[nt]
                next_state = state.transitions[s]
                for rp2 in next_state.closure:
                    if rp2.is_satisfied:
                        continue
                    s2 = rp2.next
                    if s2 not in self.lr0_rules_by_origin:
                        dr.add(s2)
                    if s2 in self.NULLABLE:
                        r.add((next_state, s2))

    def compute_includes_lookback(self):
        for nt in self.nonterminal_transitions:
            state, nonterminal = nt
            includes = []
            lookback = self.lookback[nt]
            for rp in state.closure:
                if rp.rule.origin != nonterminal:
                    continue
                state2 = state
                for i in range(rp.index, len(rp.rule.expansion)):
                    s = rp.rule.expansion[i]
                    nt2 = (state2, s)
                    state2 = state2.transitions[s]
                    if nt2 not in self.reads:
                        continue
                    for j in range(i + 1, len(rp.rule.expansion)):
                        if rp.rule.expansion[j] not in self.NULLABLE:
                            break
                    else:
                        includes.append(nt2)
                if rp.index == 0:
                    for rp2 in state2.closure:
                        if rp2.rule == rp.rule and rp2.is_satisfied:
                            lookback.add((state2, rp2.rule))
            for nt2 in includes:
                self.includes[nt2].add(nt)

    def compute_lookaheads(self):
        read_sets = digraph(self.nonterminal_transitions, self.reads, self.directly_reads)
        follow_sets = digraph(self.nonterminal_transitions, self.includes, read_sets)
        for nt, lookbacks in self.lookback.items():
            for state, rule in lookbacks:
                for s in follow_sets[nt]:
                    state.lookaheads[s].add(rule)

    def compute_lalr1_states(self) -> None:
        m: Dict[LR0ItemSet, Dict[str, Tuple]] = {}
        reduce_reduce = []
        for itemset in self.lr0_itemsets:
            actions: Dict[Symbol, Tuple] = {la: (Shift, next_state.closure) for la, next_state in itemset.transitions.items()}
            for la, rules in itemset.lookaheads.items():
                if len(rules) > 1:
                    p = [(r.options.priority or 0, r) for r in rules]
                    p.sort(key=lambda r: r[0], reverse=True)
                    best, second_best = p[:2]
                    if best[0] > second_best[0]:
                        rules = {best[1]}
                    else:
                        reduce_reduce.append((itemset, la, rules))
                        continue
                rule, = rules
                if la in actions:
                    if self.strict:
                        raise GrammarError(f'Shift/Reduce conflict for terminal {la.name}. [strict-mode]\n ')
                    elif self.debug:
                        logger.warning('Shift/Reduce conflict for terminal %s: (resolving as shift)', la.name)
                        logger.warning(' * %s', rule)
                    else:
                        logger.debug('Shift/Reduce conflict for terminal %s: (resolving as shift)', la.name)
                        logger.debug(' * %s', rule)
                else:
                    actions[la] = (Reduce, rule)
            m[itemset] = {k.name: v for k, v in actions.items()}
        if reduce_reduce:
            msgs = []
            for itemset, la, rules in reduce_reduce:
                msg = 'Reduce/Reduce collision in %s between the following rules: %s' % (la, ''.join(['\n\t- ' + str(r) for r in rules]))
                if self.debug:
                    msg += '\n    collision occurred in state: {%s\n    }' % ''.join(['\n\t' + str(x) for x in itemset.closure])
                msgs.append(msg)
            raise GrammarError('\n\n'.join(msgs))
        states = {k.closure: v for k, v in m.items()}
        end_states: Dict[str, 'State'] = {}
        for state in states:
            for rp in state:
                for start in self.lr0_start_states:
                    if rp.rule.origin.name == '$root_' + start and rp.is_satisfied:
                        assert start not in end_states
                        end_states[start] = state
        start_states = {start: state.closure for start, state in self.lr0_start_states.items()}
        _parse_table = ParseTable(states, start_states, end_states)
        if self.debug:
            self.parse_table = _parse_table
        else:
            self.parse_table = IntParseTable.from_ParseTable(_parse_table)

    def compute_lalr(self):
        self.compute_lr0_states()
        self.compute_reads_relations()
        self.compute_includes_lookback()
        self.compute_lookaheads()
        self.compute_lalr1_states()