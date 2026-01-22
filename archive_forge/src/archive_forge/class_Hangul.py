import sys
from itertools import filterfalse
from typing import List, Tuple, Union
class Hangul(unicode_set):
    """Unicode set for Hangul (Korean) Unicode Character Range"""
    _ranges: UnicodeRangeList = [(4352, 4607), (12334, 12335), (12593, 12686), (12800, 12828), (12896, 12923), (12926,), (43360, 43388), (44032, 55203), (55216, 55238), (55243, 55291), (65440, 65470), (65474, 65479), (65482, 65487), (65490, 65495), (65498, 65500)]