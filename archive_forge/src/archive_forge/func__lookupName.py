import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _lookupName(self, D={}):
    if not D:
        for n, v in getAllNamedColors().items():
            if isinstance(v, CMYKColor):
                t = (v.cyan, v.magenta, v.yellow, v.black)
                if t in D:
                    n = n + '/' + D[t]
                D[t] = n
    t = (self.cyan, self.magenta, self.yellow, self.black)
    return t in D and D[t] or None