from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _line_is_numpy_parameter_type(line_info):
    """Returns whether the line contains a numpy style parameter type definition.

  We look for a line of the form:
  x : type

  And we have to exclude false positives on argument descriptions containing a
  colon by checking the indentation of the line above.

  Args:
    line_info: Information about the current line.
  Returns:
    True if the line is a numpy parameter type definition, False otherwise.
  """
    line_stripped = line_info.remaining.strip()
    if ':' in line_stripped:
        previous_indent = line_info.previous.indentation
        current_indent = line_info.indentation
        if ':' in line_info.previous.line and current_indent > previous_indent:
            return False
        else:
            return True
    return False