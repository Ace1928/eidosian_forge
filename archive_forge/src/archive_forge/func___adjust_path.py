from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __adjust_path(self):
    """ add the current file or directory to the python path """
    path_to_append = None
    for n in range(len(self.files_or_dirs)):
        dir_name = self.__unixify(self.files_or_dirs[n])
        if os.path.isdir(dir_name):
            if not dir_name.endswith('/'):
                self.files_or_dirs[n] = dir_name + '/'
            path_to_append = os.path.normpath(dir_name)
        elif os.path.isfile(dir_name):
            path_to_append = os.path.dirname(dir_name)
        else:
            if not os.path.exists(dir_name):
                block_line = '*' * 120
                sys.stderr.write('\n%s\n* PyDev test runner error: %s does not exist.\n%s\n' % (block_line, dir_name, block_line))
                return
            msg = 'unknown type. \n%s\nshould be file or a directory.\n' % dir_name
            raise RuntimeError(msg)
    if path_to_append is not None:
        sys.path.append(path_to_append)