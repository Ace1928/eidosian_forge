import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_libdevice_paths():
    by, libdir = _get_libdevice_path_decision()
    pat = 'libdevice(\\.\\d+)*\\.bc$'
    candidates = find_file(re.compile(pat), libdir)
    out = max(candidates, default=None)
    return _env_path_tuple(by, out)