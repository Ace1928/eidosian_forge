import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_warnings_emitted_function_noargs(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertTrue(red_comet())
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(DeprecationWarning, w.category)