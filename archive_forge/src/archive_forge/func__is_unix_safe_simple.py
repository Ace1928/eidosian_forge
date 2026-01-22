import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _is_unix_safe_simple(real_path):
    if _is_unix_admin():
        return any((real_path.startswith(p) for p in _SAFE_PATHS))
    uid = os.stat(real_path).st_uid
    return uid == 0