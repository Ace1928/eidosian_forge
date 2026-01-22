from copy import deepcopy, copy
from typing import Dict, Any, Generic, List
from ..lexer import Token, LexerThread
from ..common import ParserCallbacks
from .lalr_analysis import Shift, ParseTableBase, StateT
from lark.exceptions import UnexpectedToken
class ParseConf(Generic[StateT]):
    __slots__ = ('parse_table', 'callbacks', 'start', 'start_state', 'end_state', 'states')
    parse_table: ParseTableBase[StateT]
    callbacks: ParserCallbacks
    start: str
    start_state: StateT
    end_state: StateT
    states: Dict[StateT, Dict[str, tuple]]

    def __init__(self, parse_table: ParseTableBase[StateT], callbacks: ParserCallbacks, start: str):
        self.parse_table = parse_table
        self.start_state = self.parse_table.start_states[start]
        self.end_state = self.parse_table.end_states[start]
        self.states = self.parse_table.states
        self.callbacks = callbacks
        self.start = start