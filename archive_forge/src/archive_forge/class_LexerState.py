from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
class LexerState:
    """Represents the current state of the lexer as it scans the text
    (Lexer objects are only instantiated per grammar, not per text)
    """
    __slots__ = ('text', 'line_ctr', 'last_token')
    text: str
    line_ctr: LineCounter
    last_token: Optional[Token]

    def __init__(self, text: str, line_ctr: Optional[LineCounter]=None, last_token: Optional[Token]=None):
        self.text = text
        self.line_ctr = line_ctr or LineCounter(b'\n' if isinstance(text, bytes) else '\n')
        self.last_token = last_token

    def __eq__(self, other):
        if not isinstance(other, LexerState):
            return NotImplemented
        return self.text is other.text and self.line_ctr == other.line_ctr and (self.last_token == other.last_token)

    def __copy__(self):
        return type(self)(self.text, copy(self.line_ctr), self.last_token)