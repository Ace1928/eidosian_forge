from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
class GrammarAnalyzer:

    def __init__(self, parser_conf: ParserConf, debug: bool=False, strict: bool=False):
        self.debug = debug
        self.strict = strict
        root_rules = {start: Rule(NonTerminal('$root_' + start), [NonTerminal(start), Terminal('$END')]) for start in parser_conf.start}
        rules = parser_conf.rules + list(root_rules.values())
        self.rules_by_origin: Dict[NonTerminal, List[Rule]] = classify(rules, lambda r: r.origin)
        if len(rules) != len(set(rules)):
            duplicates = [item for item, count in Counter(rules).items() if count > 1]
            raise GrammarError('Rules defined twice: %s' % ', '.join((str(i) for i in duplicates)))
        for r in rules:
            for sym in r.expansion:
                if not (sym.is_term or sym in self.rules_by_origin):
                    raise GrammarError('Using an undefined rule: %s' % sym)
        self.start_states = {start: self.expand_rule(root_rule.origin) for start, root_rule in root_rules.items()}
        self.end_states = {start: fzset({RulePtr(root_rule, len(root_rule.expansion))}) for start, root_rule in root_rules.items()}
        lr0_root_rules = {start: Rule(NonTerminal('$root_' + start), [NonTerminal(start)]) for start in parser_conf.start}
        lr0_rules = parser_conf.rules + list(lr0_root_rules.values())
        assert len(lr0_rules) == len(set(lr0_rules))
        self.lr0_rules_by_origin = classify(lr0_rules, lambda r: r.origin)
        self.lr0_start_states = {start: LR0ItemSet([RulePtr(root_rule, 0)], self.expand_rule(root_rule.origin, self.lr0_rules_by_origin)) for start, root_rule in lr0_root_rules.items()}
        self.FIRST, self.FOLLOW, self.NULLABLE = calculate_sets(rules)

    def expand_rule(self, source_rule: NonTerminal, rules_by_origin=None) -> State:
        """Returns all init_ptrs accessible by rule (recursive)"""
        if rules_by_origin is None:
            rules_by_origin = self.rules_by_origin
        init_ptrs = set()

        def _expand_rule(rule: NonTerminal) -> Iterator[NonTerminal]:
            assert not rule.is_term, rule
            for r in rules_by_origin[rule]:
                init_ptr = RulePtr(r, 0)
                init_ptrs.add(init_ptr)
                if r.expansion:
                    new_r = init_ptr.next
                    if not new_r.is_term:
                        assert isinstance(new_r, NonTerminal)
                        yield new_r
        for _ in bfs([source_rule], _expand_rule):
            pass
        return fzset(init_ptrs)