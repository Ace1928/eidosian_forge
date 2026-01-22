import os
import sys
import stat
import genericpath
from genericpath import *
def _getfinalpathname_nonstrict(path):
    allowed_winerror = (1, 2, 3, 5, 21, 32, 50, 53, 65, 67, 87, 123, 161, 1920, 1921)
    tail = path[:0]
    while path:
        try:
            path = _getfinalpathname(path)
            return join(path, tail) if tail else path
        except OSError as ex:
            if ex.winerror not in allowed_winerror:
                raise
            try:
                new_path = _readlink_deep(path)
                if new_path != path:
                    return join(new_path, tail) if tail else new_path
            except OSError:
                pass
            path, name = split(path)
            if path and (not name):
                return path + tail
            tail = join(name, tail) if tail else name
    return tail