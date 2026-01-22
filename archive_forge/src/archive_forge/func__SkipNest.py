from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
def _SkipNest(self, line, index, open_chars='[(', close_chars=')]'):
    """Skip a [...] nested bracket group starting at line[index].

    Args:
      line: The string.
      index: The starting index in string.
      open_chars: The open nesting characters.
      close_chars: The close nesting characters.

    Returns:
      The index in line after the nesting group or len(line) at end of string.
    """
    nest = 0
    while index < len(line):
        c = line[index]
        index += 1
        if c in open_chars:
            nest += 1
        elif c in close_chars:
            nest -= 1
            if nest <= 0:
                break
        elif c == self._csi_char:
            index = self._SkipControlSequence(line, index)
    return index