from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __importify(self, s, dir=False):
    """ turns directory separators into dots and removes the ".py*" extension
            so the string can be used as import statement """
    if not dir:
        dirname, fname = os.path.split(s)
        if fname.count('.') > 1:
            return
        imp_stmt_pieces = [dirname.replace('\\', '/').replace('/', '.'), os.path.splitext(fname)[0]]
        if len(imp_stmt_pieces[0]) == 0:
            imp_stmt_pieces = imp_stmt_pieces[1:]
        return '.'.join(imp_stmt_pieces)
    else:
        return s.replace('\\', '/').replace('/', '.')