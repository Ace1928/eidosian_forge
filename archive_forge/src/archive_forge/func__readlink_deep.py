import os
import sys
import stat
import genericpath
from genericpath import *
def _readlink_deep(path):
    allowed_winerror = (1, 2, 3, 5, 21, 32, 50, 67, 87, 4390, 4392, 4393)
    seen = set()
    while normcase(path) not in seen:
        seen.add(normcase(path))
        try:
            old_path = path
            path = _nt_readlink(path)
            if not isabs(path):
                if not islink(old_path):
                    path = old_path
                    break
                path = normpath(join(dirname(old_path), path))
        except OSError as ex:
            if ex.winerror in allowed_winerror:
                break
            raise
        except ValueError:
            break
    return path