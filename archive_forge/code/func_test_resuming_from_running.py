from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_resuming_from_running(self):
    self.assertTransitionAllowed(states.RUNNING, states.RESUMING)