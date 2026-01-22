import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def add_border(line):
    """Add colored borders left and right to a line."""
    new_line = border_color(config.left_border + ' ')
    new_line += line.ljust(width)[:width]
    new_line += border_color(' ' + config.right_border)
    return new_line