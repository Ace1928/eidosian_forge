import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestCaseForTesting(unittest.TestCase):

    def runTest(self):
        if test_running_hook:
            test_running_hook(self)