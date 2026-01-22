import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
@property
def __key__(self):
    """obvious way to compare colours
        Comparing across the two color models is of limited use.
        >>> cmp(CMYKColor(0,0,0,1),None)
        -1
        >>> cmp(CMYKColor(0,0,0,1),_CMYK_black)
        0
        >>> cmp(PCMYKColor(0,0,0,100),_CMYK_black)
        0
        >>> cmp(CMYKColor(0,0,0,1),Color(0,0,1)),Color(0,0,0).rgba()==CMYKColor(0,0,0,1).rgba()
        (-1, True)
        """
    return (self.cyan, self.magenta, self.yellow, self.black, self.density, self.spotName, self.alpha)