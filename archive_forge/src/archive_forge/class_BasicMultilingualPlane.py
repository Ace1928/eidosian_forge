import sys
from itertools import filterfalse
from typing import List, Tuple, Union
class BasicMultilingualPlane(unicode_set):
    """Unicode set for the Basic Multilingual Plane"""
    _ranges: UnicodeRangeList = [(32, 65535)]