from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def Split(self, line):
    """Splits a long example command line by inserting newlines.

    Args:
      line: str, The command line to split.

    Returns:
      str, The command line with newlines inserted.
    """
    lines = []
    while len(line) > self._max_index:
        before, line = self._SplitInTwo(line)
        lines.extend(before)
    lines.append(line)
    return ''.join(lines)