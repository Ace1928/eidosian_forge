from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
class MyDjangoTestSuiteRunner(DjangoTestSuiteRunner):

    def __init__(self, on_run_suite):
        DjangoTestSuiteRunner.__init__(self)
        self.on_run_suite = on_run_suite

    def build_suite(self, *args, **kwargs):
        pass

    def suite_result(self, *args, **kwargs):
        pass

    def run_suite(self, *args, **kwargs):
        self.on_run_suite()