from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _get_after_google_header(line_info):
    """Gets the remainder of the line, after a Google header."""
    colon_index = line_info.remaining.find(':')
    return line_info.remaining[colon_index + 1:]