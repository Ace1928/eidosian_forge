import collections
import os
import re
import sys
import functools
import itertools
def _follow_symlinks(filepath):
    """ In case filepath is a symlink, follow it until a
        real file is reached.
    """
    filepath = os.path.abspath(filepath)
    while os.path.islink(filepath):
        filepath = os.path.normpath(os.path.join(os.path.dirname(filepath), os.readlink(filepath)))
    return filepath