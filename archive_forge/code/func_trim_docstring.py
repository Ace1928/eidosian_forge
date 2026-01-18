from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def trim_docstring(docstring):
    """Removes indentation from triple-quoted strings.

  This is the function specified in PEP 257 to handle docstrings:
  https://www.python.org/dev/peps/pep-0257/.

  Args:
    docstring: str, a python docstring.

  Returns:
    str, docstring with indentation removed.
  """
    if not docstring:
        return ''
    max_indent = 1 << 29
    lines = docstring.expandtabs().splitlines()
    indent = max_indent
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    trimmed = [lines[0].strip()]
    if indent < max_indent:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    while trimmed and (not trimmed[-1]):
        trimmed.pop()
    while trimmed and (not trimmed[0]):
        trimmed.pop(0)
    return '\n'.join(trimmed)