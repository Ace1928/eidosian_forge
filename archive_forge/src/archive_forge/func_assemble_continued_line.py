import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def assemble_continued_line(lines, start: Tuple[int, int], end_line: int):
    """Assemble a single line from multiple continued line pieces

    Continued lines are lines ending in ``\\``, and the line following the last
    ``\\`` in the block.

    For example, this code continues over multiple lines::

        if (assign_ix is not None) \\
             and (len(line) >= assign_ix + 2) \\
             and (line[assign_ix+1].string == '%') \\
             and (line[assign_ix+2].type == tokenize.NAME):

    This statement contains four continued line pieces.
    Assembling these pieces into a single line would give::

        if (assign_ix is not None) and (len(line) >= assign_ix + 2) and (line[...

    This uses 0-indexed line numbers. *start* is (lineno, colno).

    Used to allow ``%magic`` and ``!system`` commands to be continued over
    multiple lines.
    """
    parts = [lines[start[0]][start[1]:]] + lines[start[0] + 1:end_line + 1]
    return ' '.join([p.rstrip()[:-1] for p in parts[:-1]] + [parts[-1].rstrip()])