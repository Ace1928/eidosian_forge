from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Fill(self, line):
    """Adds a line to the output, splitting to stay within the output width.

    Args:
      line: The text line.
    """
    if self._paragraph:
        self._paragraph = False
        self._out.write('<p>\n')
    self.Blank()
    if self._global_flags:
        self._global_flags = False
        line = self.LinkGlobalFlags(line)
    for word in line.split():
        n = len(word)
        if self._fill + n >= self._width:
            self._out.write('\n')
            self._fill = 0
        elif self._fill:
            self._fill += 1
            self._out.write(' ')
        self._fill += n
        self._out.write(word)