from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
class CheckTaskTransitionTest(TransitionTest):

    def setUp(self):
        super(CheckTaskTransitionTest, self).setUp()
        self.check_transition = states.check_task_transition
        self.transition_exc_regexp = '^Task transition.*not allowed'

    def test_from_pending_state(self):
        self.assertTransitions(from_state=states.PENDING, allowed=(states.RUNNING,), ignored=(states.PENDING, states.REVERTING, states.SUCCESS, states.FAILURE, states.REVERTED))

    def test_from_running_state(self):
        self.assertTransitions(from_state=states.RUNNING, allowed=(states.SUCCESS, states.FAILURE), ignored=(states.REVERTING, states.RUNNING, states.PENDING, states.REVERTED))

    def test_from_success_state(self):
        self.assertTransitions(from_state=states.SUCCESS, allowed=(states.REVERTING,), ignored=(states.RUNNING, states.SUCCESS, states.PENDING, states.FAILURE, states.REVERTED))

    def test_from_failure_state(self):
        self.assertTransitions(from_state=states.FAILURE, allowed=(states.REVERTING,), ignored=(states.FAILURE, states.RUNNING, states.PENDING, states.SUCCESS, states.REVERTED))

    def test_from_reverting_state(self):
        self.assertTransitions(from_state=states.REVERTING, allowed=(states.REVERT_FAILURE, states.REVERTED), ignored=(states.RUNNING, states.REVERTING, states.PENDING, states.SUCCESS))

    def test_from_reverted_state(self):
        self.assertTransitions(from_state=states.REVERTED, allowed=(states.PENDING,), ignored=(states.REVERTING, states.REVERTED, states.RUNNING, states.SUCCESS, states.FAILURE))