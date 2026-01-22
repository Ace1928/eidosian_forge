import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def _setup_test_suite(self, suite_name):
    """Adds a test suite to the set of suites tracked by this test run.

    Args:
      suite_name: string, The name of the test suite being initialized.
    """
    if suite_name in self.suites:
        return
    self.suites[suite_name] = []
    self.failure_counts[suite_name] = 0
    self.error_counts[suite_name] = 0