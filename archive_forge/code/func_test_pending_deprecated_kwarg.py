import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_pending_deprecated_kwarg(self):

    @removals.removed_kwarg('b', category=PendingDeprecationWarning)
    def f(b=2):
        return b
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual(3, f(b=3))
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(PendingDeprecationWarning, w.category)
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual(2, f())
    self.assertEqual(0, len(capture))