from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def assertTransitionAllowed(self, from_state, to_state):
    msg = self._DISALLOWED_TPL % (from_state, to_state)
    self.assertTrue(self.check_transition(from_state, to_state), msg=msg)