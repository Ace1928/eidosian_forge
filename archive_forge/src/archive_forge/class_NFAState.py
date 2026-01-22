from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
class NFAState:

    def __init__(self, from_rule: str):
        self.from_rule: str = from_rule
        self.arcs: List[NFAArc] = []

    def add_arc(self, next_, nonterminal_or_string=None):
        assert nonterminal_or_string is None or isinstance(nonterminal_or_string, str)
        assert isinstance(next_, NFAState)
        self.arcs.append(NFAArc(next_, nonterminal_or_string))

    def __repr__(self):
        return '<%s: from %s>' % (self.__class__.__name__, self.from_rule)