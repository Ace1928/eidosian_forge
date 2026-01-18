from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
def update_set(set1, set2):
    if not set2 or set1 > set2:
        return False
    copy = set(set1)
    set1 |= set2
    return set1 != copy