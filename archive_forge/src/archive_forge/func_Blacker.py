import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def Blacker(c, f):
    """given a color combine with black as c*f+b*(1-f) 0<=f<=1"""
    c = toColor(c)
    if isinstance(c, CMYKColorSep):
        c = c.clone()
        if isinstance(c, PCMYKColorSep):
            c.__class__ = PCMYKColor
        else:
            c.__class__ = CMYKColor
    if isinstance(c, PCMYKColor):
        b = _PCMYK_black
    elif isinstance(c, CMYKColor):
        b = _CMYK_black
    else:
        b = black
    return linearlyInterpolatedColor(b, c, 0, 1, f)