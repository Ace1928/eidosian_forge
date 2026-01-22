from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _UnFormat(self, text):
    """Returns text with all inline formatting stripped."""
    if self._csi not in text:
        return text
    stripped = []
    i = 0
    while i < len(text):
        j = text.find(self._csi, i)
        if j < 0:
            stripped.append(text[i:])
            break
        stripped.append(text[i:j])
        i = j + 2
    return ''.join(stripped)