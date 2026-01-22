import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def add_paren(self, a):
    return compose(to_format('('), indent(1, a), to_format(')'))