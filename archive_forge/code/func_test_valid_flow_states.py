from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_valid_flow_states(self):
    for start_state, end_state in states._ALLOWED_FLOW_TRANSITIONS:
        self.assertTrue(states.check_flow_transition(start_state, end_state))