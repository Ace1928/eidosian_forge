import sys
import doctest
import re
import types
from .numeric_output_checker import NumericOutputChecker
def tk_root():
    if _gui_status['tk']:
        return Tk_._default_root
    else:
        return None