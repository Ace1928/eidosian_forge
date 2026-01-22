from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
class DFAPlan:
    """
    Plans are used for the parser to create stack nodes and do the proper
    DFA state transitions.
    """

    def __init__(self, next_dfa: 'DFAState', dfa_pushes: Sequence['DFAState']=[]):
        self.next_dfa = next_dfa
        self.dfa_pushes = dfa_pushes

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.next_dfa, self.dfa_pushes)