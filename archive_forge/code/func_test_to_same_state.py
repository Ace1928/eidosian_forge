from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def test_to_same_state(self):
    self.assertTransitionIgnored(states.SUCCESS, states.SUCCESS)