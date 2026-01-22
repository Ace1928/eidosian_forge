from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def _simple_format_traceback_lines(lnum, index, lines, Colors, lvals, _line_format):
    """
    Format tracebacks lines with pointing arrow, leading numbers...

    Parameters
    ==========

    lnum: int
        number of the target line of code.
    index: int
        which line in the list should be highlighted.
    lines: list[string]
    Colors:
        ColorScheme used.
    lvals: bytes
        Values of local variables, already colored, to inject just after the error line.
    _line_format: f (str) -> (str, bool)
        return (colorized version of str, failure to do so)
    """
    numbers_width = INDENT_SIZE - 1
    res = []
    for i, line in enumerate(lines, lnum - index):
        line = py3compat.cast_unicode(line)
        new_line, err = _line_format(line, 'str')
        if not err:
            line = new_line
        if i == lnum:
            pad = numbers_width - len(str(i))
            num = '%s%s' % (debugger.make_arrow(pad), str(lnum))
            line = '%s%s%s %s%s' % (Colors.linenoEm, num, Colors.line, line, Colors.Normal)
        else:
            num = '%*s' % (numbers_width, i)
            line = '%s%s%s %s' % (Colors.lineno, num, Colors.Normal, line)
        res.append(line)
        if lvals and i == lnum:
            res.append(lvals + '\n')
    return res