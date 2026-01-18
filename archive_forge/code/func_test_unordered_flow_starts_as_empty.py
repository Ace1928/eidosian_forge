from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_starts_as_empty(self):
    f = uf.Flow('test')
    self.assertEqual(0, len(f))
    self.assertEqual([], list(f))
    self.assertEqual([], list(f.iter_links()))
    self.assertEqual(set(), f.requires)
    self.assertEqual(set(), f.provides)