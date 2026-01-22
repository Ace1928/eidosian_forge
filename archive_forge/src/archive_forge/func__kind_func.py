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
def _kind_func(string):
    if string[0] in '\'"':
        string = string[1:-1]
    if real16pattern.match(string):
        return 8
    elif real8pattern.match(string):
        return 4
    return 'kind(' + string + ')'