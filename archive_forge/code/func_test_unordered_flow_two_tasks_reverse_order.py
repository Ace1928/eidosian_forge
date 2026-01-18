from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_two_tasks_reverse_order(self):
    task1 = _task(name='task1', provides=['a'])
    task2 = _task(name='task2', requires=['a'])
    f = uf.Flow('test').add(task2).add(task1)
    self.assertEqual(2, len(f))
    self.assertEqual(set(['a']), f.requires)
    self.assertEqual(set(['a']), f.provides)