from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
class CheckRetryTransitionTest(CheckTaskTransitionTest):

    def setUp(self):
        super(CheckRetryTransitionTest, self).setUp()
        self.check_transition = states.check_retry_transition
        self.transition_exc_regexp = '^Retry transition.*not allowed'

    def test_from_success_state(self):
        self.assertTransitions(from_state=states.SUCCESS, allowed=(states.REVERTING, states.RETRYING), ignored=(states.RUNNING, states.SUCCESS, states.PENDING, states.FAILURE, states.REVERTED))

    def test_from_retrying_state(self):
        self.assertTransitions(from_state=states.RETRYING, allowed=(states.RUNNING,), ignored=(states.RETRYING, states.SUCCESS, states.PENDING, states.FAILURE, states.REVERTED))