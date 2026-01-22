import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _nvvm_lib_dir():
    if IS_WIN32:
        return ('nvvm', 'bin')
    else:
        return ('nvvm', 'lib64')