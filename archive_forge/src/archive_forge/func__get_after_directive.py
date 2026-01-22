from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _get_after_directive(line_info):
    """Gets the remainder of the line, after a directive."""
    sections = line_info.stripped.split(':', 2)
    if len(sections) > 2:
        return sections[-1]
    else:
        return ''