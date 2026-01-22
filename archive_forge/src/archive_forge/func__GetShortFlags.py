from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _GetShortFlags(flags):
    """Gets a list of single-character flags that uniquely identify a flag.

  Args:
    flags: list of strings representing flags

  Returns:
    List of single character short flags,
    where the character occurred at the start of a flag once.
  """
    short_flags = [f[0] for f in flags]
    short_flag_counts = collections.Counter(short_flags)
    return [v for v in short_flags if short_flag_counts[v] == 1]