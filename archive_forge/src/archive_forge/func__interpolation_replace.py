from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
@staticmethod
def _interpolation_replace(match, parser):
    s = match.group(1)
    if s is None:
        return match.group()
    else:
        return '%%(%s)s' % parser.optionxform(s)