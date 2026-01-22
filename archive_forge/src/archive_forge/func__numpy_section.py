from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _numpy_section(line_info):
    """Checks whether the current line is the start of a new numpy-style section.

  Numpy style sections are followed by a full line of hyphens, for example:

    Section Name
    ------------
    Section body goes here.

  Args:
    line_info: Information about the current line.
  Returns:
    A Section type if one matches, or None if no section type matches.
  """
    next_line_is_hyphens = _line_is_hyphens(line_info.next.stripped)
    if next_line_is_hyphens:
        possible_title = line_info.remaining
        return _section_from_possible_title(possible_title)
    else:
        return None