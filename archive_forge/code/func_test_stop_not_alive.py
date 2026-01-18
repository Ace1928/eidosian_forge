import threading
import time
from taskflow.engines.worker_based import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import task as task_atom
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import failure
def test_stop_not_alive(self):
    self.proxy_inst_mock.start.side_effect = None
    ex = self.executor()
    ex.start()
    ex.stop()
    self.master_mock.assert_has_calls([mock.call.proxy.start(), mock.call.proxy.wait()], any_order=True)