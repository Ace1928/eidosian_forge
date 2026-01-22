from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __is_valid_py_file(self, fname):
    """ tests that a particular file contains the proper file extension
            and is not in the list of files to exclude """
    is_valid_fname = 0
    for invalid_fname in self.__class__.__exclude_files:
        is_valid_fname += int(not fnmatch.fnmatch(fname, invalid_fname))
    if_valid_ext = 0
    for ext in self.__class__.__py_extensions:
        if_valid_ext += int(fnmatch.fnmatch(fname, ext))
    return is_valid_fname > 0 and if_valid_ext > 0