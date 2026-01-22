from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def __match_tests(self, tests, test_case, test_method_name):
    if not tests:
        return 1
    for t in tests:
        class_and_method = t.split('.')
        if len(class_and_method) == 1:
            if class_and_method[0] == test_case.__class__.__name__:
                return 1
        elif len(class_and_method) == 2:
            if class_and_method[0] == test_case.__class__.__name__ and class_and_method[1] == test_method_name:
                return 1
    return 0