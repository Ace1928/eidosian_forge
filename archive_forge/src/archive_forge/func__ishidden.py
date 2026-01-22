import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _ishidden(path):
    return path[0] in ('.', b'.'[0])