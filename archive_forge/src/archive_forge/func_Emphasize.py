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
def Emphasize(self, s, bold=True, italic=False):
    """Returns a string emphasized."""
    if self._csi:
        s = s.replace(self._csi + self._ANSI_COLOR_RESET, self._csi + self._ANSI_COLOR_RESET + self.GetFontCode(bold, italic))
    return ('{start}' + s + '{end}').format(start=self.GetFontCode(bold, italic), end=self.GetFontCode())