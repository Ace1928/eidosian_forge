import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class ResourcedTestCaseForTesting(testresources.ResourcedTestCase):

    def runTest(self):
        test_running_hook(self)