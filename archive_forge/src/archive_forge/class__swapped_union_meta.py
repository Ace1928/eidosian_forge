import sys
from ctypes import *
class _swapped_union_meta(_swapped_meta, type(Union)):
    pass