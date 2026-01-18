import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def unescape_glob(string):
    """Unescape glob pattern in `string`."""

    def unescape(s):
        for pattern in '*[]!?':
            s = s.replace('\\{0}'.format(pattern), pattern)
        return s
    return '\\'.join(map(unescape, string.split('\\\\')))