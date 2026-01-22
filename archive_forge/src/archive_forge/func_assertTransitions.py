from taskflow import exceptions as exc
from taskflow import states
from taskflow import test
def assertTransitions(self, from_state, allowed=None, ignored=None, forbidden=None):
    for a in allowed or []:
        self.assertTransitionAllowed(from_state, a)
    for i in ignored or []:
        self.assertTransitionIgnored(from_state, i)
    for f in forbidden or []:
        self.assertTransitionForbidden(from_state, f)