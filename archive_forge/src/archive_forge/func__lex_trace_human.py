from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _lex_trace_human(self, trace: str) -> T.Generator[CMakeTraceLine, None, None]:
    reg_tline = re.compile('\\s*(.*\\.(cmake|txt))\\(([0-9]+)\\):\\s*(\\w+)\\(([\\s\\S]*?) ?\\)\\s*\\n', re.MULTILINE)
    reg_other = re.compile('[^\\n]*\\n')
    loc = 0
    while loc < len(trace):
        mo_file_line = reg_tline.match(trace, loc)
        if not mo_file_line:
            skip_match = reg_other.match(trace, loc)
            if not skip_match:
                print(trace[loc:])
                raise CMakeException('Failed to parse CMake trace')
            loc = skip_match.end()
            continue
        loc = mo_file_line.end()
        file = mo_file_line.group(1)
        line = mo_file_line.group(3)
        func = mo_file_line.group(4)
        args = mo_file_line.group(5)
        argl = args.split(' ')
        argl = [a.strip() for a in argl]
        yield CMakeTraceLine(file, int(line), func, argl)