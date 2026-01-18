import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def webify(color):
    if color.startswith('calc') or color.startswith('var'):
        return color
    else:
        return '#' + color