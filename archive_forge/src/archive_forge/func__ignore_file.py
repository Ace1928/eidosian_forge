import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def _ignore_file(self, fn):
    if fn in self.ignore_paths:
        return True
    if self.ignore_hidden and os.path.basename(fn).startswith('.'):
        return True
    for pat in self.ignore_wildcards:
        if fnmatch(fn, pat):
            return True
    return False