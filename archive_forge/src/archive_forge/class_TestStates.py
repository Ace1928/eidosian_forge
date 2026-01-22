from taskflow import exceptions as excp
from taskflow import states
from taskflow import test
class TestStates(test.TestCase):

    def test_valid_flow_states(self):
        for start_state, end_state in states._ALLOWED_FLOW_TRANSITIONS:
            self.assertTrue(states.check_flow_transition(start_state, end_state))

    def test_ignored_flow_states(self):
        for start_state, end_state in states._IGNORED_FLOW_TRANSITIONS:
            self.assertFalse(states.check_flow_transition(start_state, end_state))

    def test_invalid_flow_states(self):
        invalids = [(states.RUNNING, states.PENDING), (states.REVERTED, states.RUNNING), (states.RESUMING, states.RUNNING)]
        for start_state, end_state in invalids:
            self.assertRaises(excp.InvalidState, states.check_flow_transition, start_state, end_state)

    def test_valid_job_states(self):
        for start_state, end_state in states._ALLOWED_JOB_TRANSITIONS:
            self.assertTrue(states.check_job_transition(start_state, end_state))

    def test_ignored_job_states(self):
        ignored = []
        for start_state, end_state in states._ALLOWED_JOB_TRANSITIONS:
            ignored.append((start_state, start_state))
            ignored.append((end_state, end_state))
        for start_state, end_state in ignored:
            self.assertFalse(states.check_job_transition(start_state, end_state))

    def test_invalid_job_states(self):
        invalids = [(states.COMPLETE, states.UNCLAIMED), (states.UNCLAIMED, states.COMPLETE)]
        for start_state, end_state in invalids:
            self.assertRaises(excp.InvalidState, states.check_job_transition, start_state, end_state)

    def test_valid_task_states(self):
        for start_state, end_state in states._ALLOWED_TASK_TRANSITIONS:
            self.assertTrue(states.check_task_transition(start_state, end_state))

    def test_invalid_task_states(self):
        invalids = [(states.RUNNING, states.PENDING), (states.PENDING, states.REVERTED), (states.PENDING, states.SUCCESS), (states.PENDING, states.FAILURE), (states.RETRYING, states.PENDING)]
        for start_state, end_state in invalids:
            self.assertFalse(states.check_task_transition(start_state, end_state))