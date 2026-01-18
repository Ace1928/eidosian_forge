import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_pending_warnings_emitted_class_inherit(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        s = StarLordJr('star_jr')
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(DeprecationWarning, w.category)
    self.assertEqual('star_jr', s.name)