import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
class _Chomper:

    def __init__(self, c):
        self.c = c

    def __call__(self, x, y):
        if len(y) == 0:
            return x[:-1] + [x[-1] + self.c]
        else:
            return x + [y]