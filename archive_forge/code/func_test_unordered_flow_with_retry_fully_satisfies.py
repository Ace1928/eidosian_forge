from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_with_retry_fully_satisfies(self):
    ret = retry.AlwaysRevert(provides=['b', 'a'])
    f = uf.Flow('test', ret)
    f.add(_task(name='task1', requires=['a']))
    self.assertIs(f.retry, ret)
    self.assertEqual('test_retry', ret.name)
    self.assertEqual(set([]), f.requires)
    self.assertEqual(set(['b', 'a']), f.provides)