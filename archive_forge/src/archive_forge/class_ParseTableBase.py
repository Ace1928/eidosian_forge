from typing import Dict, Set, Iterator, Tuple, List, TypeVar, Generic
from collections import defaultdict
from ..utils import classify, classify_bool, bfs, fzset, Enumerator, logger
from ..exceptions import GrammarError
from .grammar_analysis import GrammarAnalyzer, Terminal, LR0ItemSet, RulePtr, State
from ..grammar import Rule, Symbol
from ..common import ParserConf
class ParseTableBase(Generic[StateT]):
    states: Dict[StateT, Dict[str, Tuple]]
    start_states: Dict[str, StateT]
    end_states: Dict[str, StateT]

    def __init__(self, states, start_states, end_states):
        self.states = states
        self.start_states = start_states
        self.end_states = end_states

    def serialize(self, memo):
        tokens = Enumerator()
        states = {state: {tokens.get(token): (1, arg.serialize(memo)) if action is Reduce else (0, arg) for token, (action, arg) in actions.items()} for state, actions in self.states.items()}
        return {'tokens': tokens.reversed(), 'states': states, 'start_states': self.start_states, 'end_states': self.end_states}

    @classmethod
    def deserialize(cls, data, memo):
        tokens = data['tokens']
        states = {state: {tokens[token]: (Reduce, Rule.deserialize(arg, memo)) if action == 1 else (Shift, arg) for token, (action, arg) in actions.items()} for state, actions in data['states'].items()}
        return cls(states, data['start_states'], data['end_states'])