from __future__ import absolute_import, division, print_function
import re
import ast
import collections
from . import error
def _str_val(s):
    return ast.parse('u' + s).body[0].value.value