import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def Whiter(c, f):
    """given a color combine with white as c*f w*(1-f) 0<=f<=1"""
    c = toColor(c)
    if isinstance(c, CMYKColorSep):
        c = c.clone()
        if isinstance(c, PCMYKColorSep):
            c.__class__ = PCMYKColor
        else:
            c.__class__ = CMYKColor
    if isinstance(c, PCMYKColor):
        w = _PCMYK_white
    elif isinstance(c, CMYKColor):
        w = _CMYK_white
    else:
        w = white
    return linearlyInterpolatedColor(w, c, 0, 1, f)