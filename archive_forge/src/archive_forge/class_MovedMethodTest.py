import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class MovedMethodTest(test_base.TestCase):

    def test_basics(self):
        c = KittyKat()
        self.assertEqual('supermeow', c.meow())
        self.assertEqual('supermeow', c.supermeow())

    def test_warnings_emitted(self):
        c = KittyKat()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('supermeow', c.meow())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(DeprecationWarning, w.category)

    def test_warnings_emitted_pending(self):
        c = KittyKat()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('supermeow', c.maow())
        self.assertEqual(1, len(capture))
        w = capture[0]
        self.assertEqual(PendingDeprecationWarning, w.category)

    def test_warnings_not_emitted(self):
        c = KittyKat()
        with warnings.catch_warnings(record=True) as capture:
            warnings.simplefilter('always')
            self.assertEqual('supermeow', c.supermeow())
        self.assertEqual(0, len(capture))

    def test_keeps_argspec(self):
        self.assertEqual(inspect.getfullargspec(KittyKat.supermeow), inspect.getfullargspec(KittyKat.meow))