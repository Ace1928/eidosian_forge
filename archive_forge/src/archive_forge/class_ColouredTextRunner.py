from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
class ColouredTextRunner(unittest.TextTestRunner):
    """A coloured text runner which also prints failed tests on
    KeyboardInterrupt and save failed tests in a file so that they can
    be re-run.
    """
    resultclass = ColouredResult if USE_COLORS else unittest.TextTestResult

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_tnames = set()

    def _makeResult(self):
        self.result = super()._makeResult()
        return self.result

    def _write_last_failed(self):
        if self.failed_tnames:
            with open(FAILED_TESTS_FNAME, 'w') as f:
                for tname in self.failed_tnames:
                    f.write(tname + '\n')

    def _save_result(self, result):
        if not result.wasSuccessful():
            for t in result.errors + result.failures:
                tname = t[0].id()
                self.failed_tnames.add(tname)

    def _run(self, suite):
        try:
            result = super().run(suite)
        except (KeyboardInterrupt, SystemExit):
            result = self.runner.result
            result.printErrors()
            raise sys.exit(1)
        else:
            self._save_result(result)
            return result

    def _exit(self, success):
        if success:
            cprint('SUCCESS', 'green', bold=True)
            safe_rmpath(FAILED_TESTS_FNAME)
            sys.exit(0)
        else:
            cprint('FAILED', 'red', bold=True)
            self._write_last_failed()
            sys.exit(1)

    def run(self, suite):
        result = self._run(suite)
        self._exit(result.wasSuccessful())