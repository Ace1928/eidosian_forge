import math
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt
def _define_colors(self, outfile):
    colors = set()
    for _, ndef in self.style:
        if ndef['color'] is not None:
            colors.add(ndef['color'])
    for color in sorted(colors):
        outfile.write('.defcolor ' + color + ' rgb #' + color + '\n')