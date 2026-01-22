from __future__ import annotations
from dataclasses import dataclass, field
import re
import codecs
import os
import typing as T
from .mesonlib import MesonException
from . import mlog
class BlockParseException(ParseException):

    def __init__(self, text: str, line: str, lineno: int, colno: int, start_line: str, start_lineno: int, start_colno: int) -> None:
        if lineno == start_lineno:
            MesonException.__init__(self, '{}\n{}\n{}'.format(text, line, '{}^{}^'.format(' ' * start_colno, '_' * (colno - start_colno - 1))))
        else:
            MesonException.__init__(self, '%s\n%s\n%s\nFor a block that started at %d,%d\n%s\n%s' % (text, line, '%s^' % (' ' * colno), start_lineno, start_colno, start_line, '%s^' % (' ' * start_colno)))
        self.lineno = lineno
        self.colno = colno