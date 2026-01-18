import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_set_acceptable_keys_from_config(self):
    self.requireFeature(features.gpg)
    self.import_keys()
    my_gpg = gpg.GPGStrategy(FakeConfig(b'acceptable_keys=bazaar@example.com'))
    my_gpg.set_acceptable_keys(None)
    self.assertEqual(my_gpg.acceptable_keys, ['B5DEED5FCB15DAE6ECEF919587681B1EE3080E45'])