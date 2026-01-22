from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
class LR0ItemSet:
    __slots__ = ('kernel', 'closure', 'transitions', 'lookaheads')
    kernel: State
    closure: State
    transitions: Dict[Symbol, 'LR0ItemSet']
    lookaheads: Dict[Symbol, Set[Rule]]

    def __init__(self, kernel, closure):
        self.kernel = fzset(kernel)
        self.closure = fzset(closure)
        self.transitions = {}
        self.lookaheads = defaultdict(set)

    def __repr__(self):
        return '{%s | %s}' % (', '.join([repr(r) for r in self.kernel]), ', '.join([repr(r) for r in self.closure]))