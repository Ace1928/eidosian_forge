from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def Font(self, attr=None, out=None):
    """Returns the font embellishment string for attr.

    Args:
      attr: None to reset to the default font, otherwise one of renderer.BOLD,
        renderer.ITALIC, or renderer.CODE.
      out: Writes tags line to this stream if not None.

    Returns:
      The font embellishment HTML tag string.
    """
    tags = []
    if attr is None:
        for attr in (renderer.BOLD, renderer.ITALIC, renderer.CODE):
            mask = 1 << attr
            if self._font & mask:
                self._font ^= mask
                for tag in reversed(self._FONT_TAG[attr]):
                    tags.append('</%s>' % tag)
    else:
        mask = 1 << attr
        self._font ^= mask
        if self._font & mask:
            for tag in self._FONT_TAG[attr]:
                tags.append('<%s>' % tag)
        else:
            for tag in reversed(self._FONT_TAG[attr]):
                tags.append('</%s>' % tag)
    embellishment = ''.join(tags)
    if out and embellishment:
        out.write(embellishment + '\n')
    return embellishment