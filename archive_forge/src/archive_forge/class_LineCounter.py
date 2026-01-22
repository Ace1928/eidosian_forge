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
class LineCounter:
    """A utility class for keeping track of line & column information"""
    __slots__ = ('char_pos', 'line', 'column', 'line_start_pos', 'newline_char')

    def __init__(self, newline_char):
        self.newline_char = newline_char
        self.char_pos = 0
        self.line = 1
        self.column = 1
        self.line_start_pos = 0

    def __eq__(self, other):
        if not isinstance(other, LineCounter):
            return NotImplemented
        return self.char_pos == other.char_pos and self.newline_char == other.newline_char

    def feed(self, token: Token, test_newline=True):
        """Consume a token and calculate the new line & column.

        As an optional optimization, set test_newline=False if token doesn't contain a newline.
        """
        if test_newline:
            newlines = token.count(self.newline_char)
            if newlines:
                self.line += newlines
                self.line_start_pos = self.char_pos + token.rindex(self.newline_char) + 1
        self.char_pos += len(token)
        self.column = self.char_pos - self.line_start_pos + 1