import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class StopPPException(Exception):

    def __str__(self):
        return 'pp-interrupted'