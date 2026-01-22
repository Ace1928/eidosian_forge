from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
class TransitionTest(test.TestCase):
    _DISALLOWED_TPL = "Transition from '%s' to '%s' was found to be disallowed"
    _NOT_IGNORED_TPL = "Transition from '%s' to '%s' was not ignored"

    def assertTransitionAllowed(self, from_state, to_state):
        msg = self._DISALLOWED_TPL % (from_state, to_state)
        self.assertTrue(self.check_transition(from_state, to_state), msg=msg)

    def assertTransitionIgnored(self, from_state, to_state):
        msg = self._NOT_IGNORED_TPL % (from_state, to_state)
        self.assertFalse(self.check_transition(from_state, to_state), msg=msg)

    def assertTransitionForbidden(self, from_state, to_state):
        self.assertRaisesRegex(exc.InvalidState, self.transition_exc_regexp, self.check_transition, from_state, to_state)

    def assertTransitions(self, from_state, allowed=None, ignored=None, forbidden=None):
        for a in allowed or []:
            self.assertTransitionAllowed(from_state, a)
        for i in ignored or []:
            self.assertTransitionIgnored(from_state, i)
        for f in forbidden or []:
            self.assertTransitionForbidden(from_state, f)