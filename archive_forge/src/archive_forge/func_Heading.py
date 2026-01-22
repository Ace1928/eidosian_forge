from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Heading(self, level, heading):
    """Renders a heading.

    Args:
      level: The heading level counting from 1.
      heading: The heading text.
    """
    if level == 1 and heading.endswith('(1)'):
        return
    self._Flush()
    self.Font(out=self._out)
    self.List(0)
    if self._heading:
        self._out.write(self._heading)
    self._Heading(level, heading)
    self._section = True
    if heading.endswith(' WIDE FLAGS'):
        self._global_flags = True