import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_warnings_not_emitted(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual((1, 2), blip_blop(blop=2))
    self.assertEqual(0, len(capture))
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual(2, blip_blop_3(blop=2))
    self.assertEqual(0, len(capture))