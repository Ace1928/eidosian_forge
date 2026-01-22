import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def _color_tuple(self, color):
    try:
        rgb = int(str(color), 16)
    except ValueError:
        return None
    r = rgb >> 16 & 255
    g = rgb >> 8 & 255
    b = rgb & 255
    return (r, g, b)