import taskflow.engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_other_name(self):
    task = base.FunctorTask(add, name='my task')
    self.assertEqual('my task', task.name)