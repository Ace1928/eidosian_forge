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
def SplitLine(self, line, width):
    """Splits line into width length chunks.

    Args:
      line: The line to split.
      width: The width of each chunk except the last which could be smaller than
        width.

    Returns:
      A list of chunks, all but the last with display width == width.
    """
    lines = []
    chunk = ''
    w = 0
    keep = False
    for normal, control in self.SplitIntoNormalAndControl(line):
        keep = True
        while True:
            n = width - w
            w += len(normal)
            if w <= width:
                break
            lines.append(chunk + normal[:n])
            chunk = ''
            keep = False
            w = 0
            normal = normal[n:]
        chunk += normal + control
    if chunk or keep:
        lines.append(chunk)
    return lines