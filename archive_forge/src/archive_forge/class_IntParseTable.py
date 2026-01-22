from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
class IntParseTable(ParseTableBase[int]):
    """Parse-table whose key is int. Best for performance."""

    @classmethod
    def from_ParseTable(cls, parse_table: ParseTable):
        enum = list(parse_table.states)
        state_to_idx: Dict['State', int] = {s: i for i, s in enumerate(enum)}
        int_states = {}
        for s, la in parse_table.states.items():
            la = {k: (v[0], state_to_idx[v[1]]) if v[0] is Shift else v for k, v in la.items()}
            int_states[state_to_idx[s]] = la
        start_states = {start: state_to_idx[s] for start, s in parse_table.start_states.items()}
        end_states = {start: state_to_idx[s] for start, s in parse_table.end_states.items()}
        return cls(int_states, start_states, end_states)