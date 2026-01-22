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
class Colorizer(object):
    """Resource string colorizer.

  Attributes:
    _con: ConsoleAttr object.
    _color: Color name.
    _string: The string to colorize.
    _justify: The justification function, no justification if None. For example,
      justify=lambda s: s.center(10)
  """

    def __init__(self, string, color, justify=None):
        """Constructor.

    Args:
      string: The string to colorize.
      color: Color name used to index ConsoleAttr._ANSI_COLOR.
      justify: The justification function, no justification if None. For
        example, justify=lambda s: s.center(10)
    """
        self._con = GetConsoleAttr()
        self._color = color
        self._string = string
        self._justify = justify

    def __eq__(self, other):
        return self._string == six.text_type(other)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return self._string > six.text_type(other)

    def __lt__(self, other):
        return self._string < six.text_type(other)

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other

    def __len__(self):
        return self._con.DisplayWidth(self._string)

    def __str__(self):
        return self._string

    def Render(self, stream, justify=None):
        """Renders the string as self._color on the console.

    Args:
      stream: The stream to render the string to. The stream given here *must*
        have the same encoding as sys.stdout for this to work properly.
      justify: The justification function, self._justify if None.
    """
        stream.write(self._con.Colorize(self._string, self._color, justify or self._justify))