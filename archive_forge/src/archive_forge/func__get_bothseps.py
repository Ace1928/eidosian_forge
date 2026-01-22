import os
import sys
import stat
import genericpath
from genericpath import *
def _get_bothseps(path):
    if isinstance(path, bytes):
        return b'\\/'
    else:
        return '\\/'