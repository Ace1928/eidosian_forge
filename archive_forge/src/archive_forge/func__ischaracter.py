import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
def _ischaracter(var):
    return 'typespec' in var and var['typespec'] == 'character' and (not isexternal(var))