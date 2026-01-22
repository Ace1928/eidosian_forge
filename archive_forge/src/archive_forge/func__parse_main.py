import logging
import re
from typing import (
from . import settings
from .utils import choplist
def _parse_main(self, s: bytes, i: int) -> int:
    m = NONSPC.search(s, i)
    if not m:
        return len(s)
    j = m.start(0)
    c = s[j:j + 1]
    self._curtokenpos = self.bufpos + j
    if c == b'%':
        self._curtoken = b'%'
        self._parse1 = self._parse_comment
        return j + 1
    elif c == b'/':
        self._curtoken = b''
        self._parse1 = self._parse_literal
        return j + 1
    elif c in b'-+' or c.isdigit():
        self._curtoken = c
        self._parse1 = self._parse_number
        return j + 1
    elif c == b'.':
        self._curtoken = c
        self._parse1 = self._parse_float
        return j + 1
    elif c.isalpha():
        self._curtoken = c
        self._parse1 = self._parse_keyword
        return j + 1
    elif c == b'(':
        self._curtoken = b''
        self.paren = 1
        self._parse1 = self._parse_string
        return j + 1
    elif c == b'<':
        self._curtoken = b''
        self._parse1 = self._parse_wopen
        return j + 1
    elif c == b'>':
        self._curtoken = b''
        self._parse1 = self._parse_wclose
        return j + 1
    elif c == b'\x00':
        return j + 1
    else:
        self._add_token(KWD(c))
        return j + 1