import testtools
import testresources
from testresources import split_by_resources, _resource_graph
from testresources.tests import ResultWithResourceExtensions
import unittest
def test_overlapping(self):
    res1 = testresources.TestResourceManager()
    res2 = testresources.TestResourceManager()
    resset1 = frozenset([res1])
    resset2 = frozenset([res2])
    resset3 = frozenset([res1, res2])
    resource_sets = [resset1, resset2, resset3]
    result = _resource_graph(resource_sets)
    self.assertEqual({resset1: set([resset3]), resset2: set([resset3]), resset3: set([resset1, resset2])}, result)