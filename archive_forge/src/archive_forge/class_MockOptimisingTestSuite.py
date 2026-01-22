import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class MockOptimisingTestSuite(testresources.OptimisingTestSuite):

    def sortTests(self):
        self.sorted = True