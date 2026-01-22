from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __unixify(self, s):
    """ stupid windows. converts the backslash to forwardslash for consistency """
    return os.path.normpath(s).replace(os.sep, '/')