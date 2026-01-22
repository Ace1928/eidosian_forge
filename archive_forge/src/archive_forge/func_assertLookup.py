import sys
from breezy import tests
from breezy.tests import features
def assertLookup(self, offset, value, obj, key):
    self.assertEqual((offset, value), obj._test_lookup(key))