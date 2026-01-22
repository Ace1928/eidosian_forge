from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def addclosure(nfa_state, base_nfa_set):
    assert isinstance(nfa_state, NFAState)
    if nfa_state in base_nfa_set:
        return
    base_nfa_set.add(nfa_state)
    for nfa_arc in nfa_state.arcs:
        if nfa_arc.nonterminal_or_string is None:
            addclosure(nfa_arc.next, base_nfa_set)