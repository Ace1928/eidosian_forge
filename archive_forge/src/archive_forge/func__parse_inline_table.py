from __future__ import annotations
import datetime
import re
import string
from tomlkit._compat import decode
from tomlkit._utils import RFC_3339_LOOSE
from tomlkit._utils import _escaped
from tomlkit._utils import parse_rfc3339
from tomlkit.container import Container
from tomlkit.exceptions import EmptyKeyError
from tomlkit.exceptions import EmptyTableNameError
from tomlkit.exceptions import InternalParserError
from tomlkit.exceptions import InvalidCharInStringError
from tomlkit.exceptions import InvalidControlChar
from tomlkit.exceptions import InvalidDateError
from tomlkit.exceptions import InvalidDateTimeError
from tomlkit.exceptions import InvalidNumberError
from tomlkit.exceptions import InvalidTimeError
from tomlkit.exceptions import InvalidUnicodeValueError
from tomlkit.exceptions import ParseError
from tomlkit.exceptions import UnexpectedCharError
from tomlkit.exceptions import UnexpectedEofError
from tomlkit.items import AoT
from tomlkit.items import Array
from tomlkit.items import Bool
from tomlkit.items import BoolType
from tomlkit.items import Comment
from tomlkit.items import Date
from tomlkit.items import DateTime
from tomlkit.items import Float
from tomlkit.items import InlineTable
from tomlkit.items import Integer
from tomlkit.items import Item
from tomlkit.items import Key
from tomlkit.items import KeyType
from tomlkit.items import Null
from tomlkit.items import SingleKey
from tomlkit.items import String
from tomlkit.items import StringType
from tomlkit.items import Table
from tomlkit.items import Time
from tomlkit.items import Trivia
from tomlkit.items import Whitespace
from tomlkit.source import Source
from tomlkit.toml_char import TOMLChar
from tomlkit.toml_document import TOMLDocument
def _parse_inline_table(self) -> InlineTable:
    self.inc(exception=UnexpectedEofError)
    elems = Container(True)
    trailing_comma = None
    while True:
        mark = self._idx
        self.consume(TOMLChar.SPACES)
        raw = self._src[mark:self._idx]
        if raw:
            elems.add(Whitespace(raw))
        if not trailing_comma:
            if self._current == '}':
                self.inc()
                break
            if trailing_comma is False or (trailing_comma is None and self._current == ','):
                raise self.parse_error(UnexpectedCharError, self._current)
        elif self._current == '}' or self._current == ',':
            raise self.parse_error(UnexpectedCharError, self._current)
        key, val = self._parse_key_value(False)
        elems.add(key, val)
        mark = self._idx
        self.consume(TOMLChar.SPACES)
        raw = self._src[mark:self._idx]
        if raw:
            elems.add(Whitespace(raw))
        trailing_comma = self._current == ','
        if trailing_comma:
            self.inc(exception=UnexpectedEofError)
    return InlineTable(elems, Trivia())