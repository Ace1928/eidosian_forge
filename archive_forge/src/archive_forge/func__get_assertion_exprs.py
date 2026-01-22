import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
@functools.lru_cache(maxsize=1)
def _get_assertion_exprs(src: bytes) -> Dict[int, str]:
    """Return a mapping from {lineno: "assertion test expression"}."""
    ret: Dict[int, str] = {}
    depth = 0
    lines: List[str] = []
    assert_lineno: Optional[int] = None
    seen_lines: Set[int] = set()

    def _write_and_reset() -> None:
        nonlocal depth, lines, assert_lineno, seen_lines
        assert assert_lineno is not None
        ret[assert_lineno] = ''.join(lines).rstrip().rstrip('\\')
        depth = 0
        lines = []
        assert_lineno = None
        seen_lines = set()
    tokens = tokenize.tokenize(io.BytesIO(src).readline)
    for tp, source, (lineno, offset), _, line in tokens:
        if tp == tokenize.NAME and source == 'assert':
            assert_lineno = lineno
        elif assert_lineno is not None:
            if tp == tokenize.OP and source in '([{':
                depth += 1
            elif tp == tokenize.OP and source in ')]}':
                depth -= 1
            if not lines:
                lines.append(line[offset:])
                seen_lines.add(lineno)
            elif depth == 0 and tp == tokenize.OP and (source == ','):
                if lineno in seen_lines and len(lines) == 1:
                    offset_in_trimmed = offset + len(lines[-1]) - len(line)
                    lines[-1] = lines[-1][:offset_in_trimmed]
                elif lineno in seen_lines:
                    lines[-1] = lines[-1][:offset]
                else:
                    lines.append(line[:offset])
                _write_and_reset()
            elif tp in {tokenize.NEWLINE, tokenize.ENDMARKER}:
                _write_and_reset()
            elif lines and lineno not in seen_lines:
                lines.append(line)
                seen_lines.add(lineno)
    return ret