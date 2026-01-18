from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def test_targeted_flow_one_node(self):
    f = gf.TargetedFlow('test')
    task1 = _task('task1', provides=['a'], requires=[])
    f.add(task1)
    f.set_target(task1)
    self.assertEqual(1, len(f))
    self.assertCountEqual(f, [task1])