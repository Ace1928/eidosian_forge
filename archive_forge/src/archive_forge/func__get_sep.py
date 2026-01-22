import os
import sys
import stat
import genericpath
from genericpath import *
def _get_sep(path):
    if isinstance(path, bytes):
        return b'/'
    else:
        return '/'