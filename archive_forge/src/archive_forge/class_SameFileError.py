import os
import sys
import stat
import fnmatch
import collections
import errno
class SameFileError(Error):
    """Raised when source and destination are the same file."""