import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _text_width(s):
    return sum((2 if east_asian_width(ch) in 'FW' else 1 for ch in str(s)))