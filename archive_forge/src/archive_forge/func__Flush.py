from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def _Flush(self):
    """Flushes the current collection of Fill() lines."""
    self._paragraph = False
    if self._fill:
        self._section = False
        if self._example:
            self._example = False
            self._out.write('</code>\n')
        self._fill = 0
        self._out.write('\n')
        self.Content()