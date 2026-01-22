import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def _psPageSetupStr(self, pageheight, initialColor, font_family, font_size, line_width):
    """ps code for settin up coordinate system for page in accords w/ piddle standards"""
    r, g, b = (initialColor.red, initialColor.green, initialColor.blue)
    return '\n%% initialize\n\n2 setlinecap\n\n0 %d\ntranslate\n\n%s %s %s setrgbcolor\n(%s) findfont %s scalefont setfont\n%s setlinewidth' % (pageheight, r, g, b, font_family, font_size, line_width)