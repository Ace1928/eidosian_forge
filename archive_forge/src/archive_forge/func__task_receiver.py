import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def _task_receiver(self, state, details):
    super(SuspendingListener, self)._task_receiver(state, details)
    if (details['task_name'], state) == self._revert_match:
        self._engine.suspend()