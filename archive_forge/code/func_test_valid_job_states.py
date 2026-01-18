from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
def test_valid_job_states(self):
    for start_state, end_state in states._ALLOWED_JOB_TRANSITIONS:
        self.assertTrue(states.check_job_transition(start_state, end_state))