from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_unordered_flow_two_task_same_provide(self):
    task1 = _task(name='task1', provides=['a', 'b'])
    task2 = _task(name='task2', provides=['a', 'c'])
    f = uf.Flow('test')
    f.add(task2, task1)
    self.assertEqual(2, len(f))