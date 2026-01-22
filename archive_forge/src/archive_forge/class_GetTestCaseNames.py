from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
class GetTestCaseNames:
    """Yes, we need a class for that (cannot use outer context on jython 2.1)"""

    def __init__(self, accepted_classes, accepted_methods):
        self.accepted_classes = accepted_classes
        self.accepted_methods = accepted_methods

    def __call__(self, testCaseClass):
        """Return a sorted sequence of method names found within testCaseClass"""
        testFnNames = []
        className = testCaseClass.__name__
        if className in self.accepted_classes:
            for attrname in dir(testCaseClass):
                if attrname.startswith('test') and hasattr(getattr(testCaseClass, attrname), '__call__'):
                    testFnNames.append(attrname)
        else:
            for attrname in dir(testCaseClass):
                if className + '.' + attrname in self.accepted_methods:
                    if hasattr(getattr(testCaseClass, attrname), '__call__'):
                        testFnNames.append(attrname)
        testFnNames.sort()
        return testFnNames