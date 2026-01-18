from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_recache_on_add(self):
    f = gf.TargetedFlow('test')
    task1 = _task('task1', provides=[], requires=['a'])
    f.add(task1)
    f.set_target(task1)
    self.assertEqual(1, len(f))
    task2 = _task('task2', provides=['a'], requires=[])
    f.add(task2)
    self.assertEqual(2, len(f))