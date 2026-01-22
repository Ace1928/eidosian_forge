import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _StandardizePath(self, path):
    """Do :standardizepath processing for path."""
    if '/' in path:
        prefix, rest = ('', path)
        if path.startswith('@'):
            prefix, rest = path.split('/', 1)
        rest = os.path.normpath(rest)
        path = os.path.join(prefix, rest)
    return path