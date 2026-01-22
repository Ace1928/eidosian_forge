from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _get_directive(line_info):
    """Gets a directive from the start of the line.

  If the line is ":param str foo: Description of foo", then
  _get_directive(line_info) returns "param str foo".

  Args:
    line_info: Information about the current line.
  Returns:
    The contents of a directive, or None if the line doesn't start with a
    directive.
  """
    if line_info.stripped.startswith(':'):
        return line_info.stripped.split(':', 2)[1]
    else:
        return None