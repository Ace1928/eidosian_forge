from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Escape(self, buf):
    """Escapes special characters in normal text.

    This is applied before font embellishments.

    Args:
      buf: Normal text that may contain special characters.

    Returns:
      The escaped string.
    """
    return ''.join((self._ESCAPE.get(c, c) for c in buf))