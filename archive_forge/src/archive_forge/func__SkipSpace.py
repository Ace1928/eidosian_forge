from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
def _SkipSpace(self, line, index):
    """Skip space characters starting at line[index].

    Args:
      line: The string.
      index: The starting index in string.

    Returns:
      The index in line after spaces or len(line) at end of string.
    """
    while index < len(line):
        c = line[index]
        if c != ' ':
            break
        index += 1
    return index