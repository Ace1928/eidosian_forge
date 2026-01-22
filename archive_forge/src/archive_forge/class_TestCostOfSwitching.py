import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class TestCostOfSwitching(testtools.TestCase):
    """Tests for cost_of_switching."""

    def setUp(self):
        super(TestCostOfSwitching, self).setUp()
        self.suite = testresources.OptimisingTestSuite()

    def makeResource(self, setUpCost=1, tearDownCost=1):
        resource = testresources.TestResource()
        resource.setUpCost = setUpCost
        resource.tearDownCost = tearDownCost
        return resource

    def testNoResources(self):
        self.assertEqual(0, self.suite.cost_of_switching(set(), set()))

    def testSameResources(self):
        a = self.makeResource()
        b = self.makeResource()
        self.assertEqual(0, self.suite.cost_of_switching(set([a]), set([a])))
        self.assertEqual(0, self.suite.cost_of_switching(set([a, b]), set([a, b])))

    def testNewResources(self):
        a = self.makeResource()
        b = self.makeResource()
        self.assertEqual(1, self.suite.cost_of_switching(set(), set([a])))
        self.assertEqual(1, self.suite.cost_of_switching(set([a]), set([a, b])))
        self.assertEqual(2, self.suite.cost_of_switching(set(), set([a, b])))

    def testOldResources(self):
        a = self.makeResource()
        b = self.makeResource()
        self.assertEqual(1, self.suite.cost_of_switching(set([a]), set()))
        self.assertEqual(1, self.suite.cost_of_switching(set([a, b]), set([a])))
        self.assertEqual(2, self.suite.cost_of_switching(set([a, b]), set()))

    def testCombo(self):
        a = self.makeResource()
        b = self.makeResource()
        c = self.makeResource()
        self.assertEqual(2, self.suite.cost_of_switching(set([a]), set([b])))
        self.assertEqual(2, self.suite.cost_of_switching(set([a, c]), set([b, c])))