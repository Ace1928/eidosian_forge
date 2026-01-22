import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def _eval_scalar(value, params):
    if _is_kind_number(value):
        value = value.split('_')[0]
    try:
        value = eval(value, {}, params)
        value = (repr if isinstance(value, str) else str)(value)
    except (NameError, SyntaxError, TypeError):
        return value
    except Exception as msg:
        errmess('"%s" in evaluating %r (available names: %s)\n' % (msg, value, list(params.keys())))
    return value