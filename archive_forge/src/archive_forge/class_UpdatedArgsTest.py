import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class UpdatedArgsTest(test_base.TestCase):

    def test_basic(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('The cat meowed quietly', blip_blop_blip())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(FutureWarning, w.category)

    def test_kwarg_set(self):
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('The kitten meowed quietly', blip_blop_blip(type='kitten'))
        self.assertEqual(0, len(capture))

    def test_argspec_preserved(self):
        self.assertEqual(inspect.getfullargspec(blip_blop_blip_unwrapped), inspect.getfullargspec(blip_blop_blip))