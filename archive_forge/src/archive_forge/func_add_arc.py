from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def add_arc(self, next_, label):
    assert isinstance(label, str)
    assert label not in self.arcs
    assert isinstance(next_, DFAState)
    self.arcs[label] = next_