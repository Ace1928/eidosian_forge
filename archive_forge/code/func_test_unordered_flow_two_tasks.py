from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_two_tasks(self):
    task1 = _task(name='task1')
    task2 = _task(name='task2')
    f = uf.Flow('test').add(task1, task2)
    self.assertEqual(2, len(f))
    self.assertEqual(set([task1, task2]), set(f))
    self.assertEqual([], list(f.iter_links()))