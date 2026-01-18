from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint as ep
from taskflow import task
from taskflow import test
from taskflow.tests import utils
def test_to_str(self):
    self.assertEqual(self.task_cls_name, str(self.task_ep))