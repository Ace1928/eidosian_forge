import textwrap
from unicodedata import east_asian_width as _eawidth
from . import osutils
def _width(self, s):
    """Returns width for s.

        When s is unicode, take care of east asian width.
        When s is bytes, treat all byte is single width character.
        """
    charwidth = self._unicode_char_width
    return sum((charwidth(c) for c in s))