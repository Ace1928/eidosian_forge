import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_set_acceptable_keys_unknown(self):
    self.requireFeature(features.gpg)
    my_gpg = gpg.GPGStrategy(FakeConfig())
    self.notes = []

    def note(*args):
        self.notes.append(args[0] % args[1:])
    self.overrideAttr(trace, 'note', note)
    my_gpg.set_acceptable_keys('unknown')
    self.assertEqual(my_gpg.acceptable_keys, [])
    self.assertEqual(self.notes, ['No GnuPG key results for pattern: unknown'])