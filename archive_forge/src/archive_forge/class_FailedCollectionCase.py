import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class FailedCollectionCase(unittest.TestCase):
    """Pseudo-test to run and report failure if given case was uncollected"""

    def __init__(self, case):
        super().__init__('fail_uncollected')
        self._problem_case_id = case.id()

    def id(self):
        if self._problem_case_id[-1:] == ')':
            return self._problem_case_id[:-1] + ',uncollected)'
        return self._problem_case_id + '(uncollected)'

    def fail_uncollected(self):
        self.fail('Uncollected test case: ' + self._problem_case_id)