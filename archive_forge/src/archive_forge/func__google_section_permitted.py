from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _google_section_permitted(line_info, state):
    """Returns whether a new google section is permitted to start here.

  Q: Why might a new Google section not be allowed?
  A: If we're in the middle of a Google "Args" section, then lines that start
  "param:" will usually be a new arg, rather than a new section.
  We use whitespace to determine when the Args section has actually ended.

  A Google section ends when either:
  - A new google section begins at either
    - indentation less than indentation of line 1 of the previous section
    - or <= indentation of the previous section
  - Or the docstring terminates.

  Args:
    line_info: Information about the current line.
    state: The state of the parser.
  Returns:
    True or False, indicating whether a new Google section is permitted at the
    current line.
  """
    if state.section.indentation is None:
        return True
    return line_info.indentation <= state.section.indentation or line_info.indentation < state.section.line1_indentation