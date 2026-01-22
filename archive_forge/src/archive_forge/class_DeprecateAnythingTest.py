import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class DeprecateAnythingTest(test_base.TestCase):

    def test_generation(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            debtcollector.deprecate('Its broken')
            debtcollector.deprecate('Its really broken')
        self.assertEqual(2, len(capture))