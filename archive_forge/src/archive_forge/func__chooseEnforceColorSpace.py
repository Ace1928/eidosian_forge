import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def _chooseEnforceColorSpace(enforceColorSpace):
    if enforceColorSpace is not None and (not callable(enforceColorSpace)):
        if isinstance(enforceColorSpace, str):
            enforceColorSpace = enforceColorSpace.upper()
        if enforceColorSpace == 'CMYK':
            enforceColorSpace = _enforceCMYK
        elif enforceColorSpace == 'RGB':
            enforceColorSpace = _enforceRGB
        elif enforceColorSpace == 'SEP':
            enforceColorSpace = _enforceSEP
        elif enforceColorSpace == 'SEP_BLACK':
            enforceColorSpace = _enforceSEP_BLACK
        elif enforceColorSpace == 'SEP_CMYK':
            enforceColorSpace = _enforceSEP_CMYK
        else:
            raise ValueError('Invalid value for Canvas argument enforceColorSpace=%r' % enforceColorSpace)
    return enforceColorSpace