import os
import sys
import stat
import fnmatch
import collections
import errno
def _nop(*args, ns=None, follow_symlinks=None):
    pass