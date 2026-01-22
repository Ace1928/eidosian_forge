from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
def SplitIntoNormalAndControl(self, buf):
    """Returns a list of (normal_string, control_sequence) tuples from buf.

    Args:
      buf: The input string containing one or more control sequences
        interspersed with normal strings.

    Returns:
      A list of (normal_string, control_sequence) tuples.
    """
    if not self._csi or not buf:
        return [(buf, '')]
    seq = []
    i = 0
    while i < len(buf):
        c = buf.find(self._csi, i)
        if c < 0:
            seq.append((buf[i:], ''))
            break
        normal = buf[i:c]
        i = c + self.GetControlSequenceLen(buf[c:])
        seq.append((normal, buf[c:i]))
    return seq