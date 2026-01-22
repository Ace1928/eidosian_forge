import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
@memoize
def RelativePath(path, relative_to, follow_path_symlink=True):
    if follow_path_symlink:
        path = os.path.realpath(path)
    else:
        path = os.path.abspath(path)
    relative_to = os.path.realpath(relative_to)
    if sys.platform == 'win32':
        if os.path.splitdrive(path)[0].lower() != os.path.splitdrive(relative_to)[0].lower():
            return path
    path_split = path.split(os.path.sep)
    relative_to_split = relative_to.split(os.path.sep)
    prefix_len = len(os.path.commonprefix([path_split, relative_to_split]))
    relative_split = [os.path.pardir] * (len(relative_to_split) - prefix_len) + path_split[prefix_len:]
    if len(relative_split) == 0:
        return ''
    return os.path.join(*relative_split)