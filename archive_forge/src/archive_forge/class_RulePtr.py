from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
class RulePtr:
    __slots__ = ('rule', 'index')
    rule: Rule
    index: int

    def __init__(self, rule: Rule, index: int):
        assert isinstance(rule, Rule)
        assert index <= len(rule.expansion)
        self.rule = rule
        self.index = index

    def __repr__(self):
        before = [x.name for x in self.rule.expansion[:self.index]]
        after = [x.name for x in self.rule.expansion[self.index:]]
        return '<%s : %s * %s>' % (self.rule.origin.name, ' '.join(before), ' '.join(after))

    @property
    def next(self) -> Symbol:
        return self.rule.expansion[self.index]

    def advance(self, sym: Symbol) -> 'RulePtr':
        assert self.next == sym
        return RulePtr(self.rule, self.index + 1)

    @property
    def is_satisfied(self) -> bool:
        return self.index == len(self.rule.expansion)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RulePtr):
            return NotImplemented
        return self.rule == other.rule and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.rule, self.index))