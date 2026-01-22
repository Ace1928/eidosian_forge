import os
import sys
import stat
import fnmatch
import collections
import errno
def _rmtree_unsafe(path, onerror):
    try:
        with os.scandir(path) as scandir_it:
            entries = list(scandir_it)
    except OSError:
        onerror(os.scandir, path, sys.exc_info())
        entries = []
    for entry in entries:
        fullname = entry.path
        if _rmtree_isdir(entry):
            try:
                if entry.is_symlink():
                    raise OSError('Cannot call rmtree on a symbolic link')
            except OSError:
                onerror(os.path.islink, fullname, sys.exc_info())
                continue
            _rmtree_unsafe(fullname, onerror)
        else:
            try:
                os.unlink(fullname)
            except OSError:
                onerror(os.unlink, fullname, sys.exc_info())
    try:
        os.rmdir(path)
    except OSError:
        onerror(os.rmdir, path, sys.exc_info())