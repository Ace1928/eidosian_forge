import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def toColorOrNone(arg, default=None):
    """as above but allows None as a legal value"""
    if arg is None:
        return None
    else:
        return toColor(arg, default)