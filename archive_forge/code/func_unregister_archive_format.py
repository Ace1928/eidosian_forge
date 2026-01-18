import os
import sys
import stat
import fnmatch
import collections
import errno
def unregister_archive_format(name):
    del _ARCHIVE_FORMATS[name]