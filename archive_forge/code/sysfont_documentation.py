import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
pygame.font.match_font(name, bold=0, italic=0) -> name
    find the filename for the named system font

    This performs the same font search as the SysFont()
    function, only it returns the path to the TTF file
    that would be loaded. The font name can also be an
    iterable of font names or a string/bytes of comma-separated
    font names to try.

    If no match is found, None is returned.
    