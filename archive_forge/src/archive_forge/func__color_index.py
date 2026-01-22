import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def _color_index(self, color):
    index = self.best_match.get(color, None)
    if color in ansicolors:
        index = color
        self.best_match[color] = index
    if index is None:
        try:
            rgb = int(str(color), 16)
        except ValueError:
            rgb = 0
        r = rgb >> 16 & 255
        g = rgb >> 8 & 255
        b = rgb & 255
        index = self._closest_color(r, g, b)
        self.best_match[color] = index
    return index