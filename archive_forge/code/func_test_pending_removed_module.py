import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_pending_removed_module(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        removals.removed_module(__name__, category=PendingDeprecationWarning)
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(PendingDeprecationWarning, w.category)