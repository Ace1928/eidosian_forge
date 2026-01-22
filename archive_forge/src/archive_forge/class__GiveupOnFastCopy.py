import os
import sys
import stat
import fnmatch
import collections
import errno
class _GiveupOnFastCopy(Exception):
    """Raised as a signal to fallback on using raw read()/write()
    file copy when fast-copy functions fail to do so.
    """