from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _create_line_info(line, next_line, previous_line):
    """Returns information about the current line and surrounding lines."""
    line_info = Namespace()
    line_info.line = line
    line_info.stripped = line.strip()
    line_info.remaining_raw = line_info.line
    line_info.remaining = line_info.stripped
    line_info.indentation = len(line) - len(line.lstrip())
    line_info.next.line = next_line
    next_line_exists = next_line is not None
    line_info.next.stripped = next_line.strip() if next_line_exists else None
    line_info.next.indentation = len(next_line) - len(next_line.lstrip()) if next_line_exists else None
    line_info.previous.line = previous_line
    previous_line_exists = previous_line is not None
    line_info.previous.indentation = len(previous_line) - len(previous_line.lstrip()) if previous_line_exists else None
    return line_info