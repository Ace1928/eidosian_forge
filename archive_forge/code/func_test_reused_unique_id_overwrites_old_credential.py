import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_reused_unique_id_overwrites_old_credential(self):
    with fake_keyring(self.keyring):
        credential1 = self.make_credential('consumer key1')
        self.store.save(credential1, 'the only key')
        credential2 = self.make_credential('consumer key2')
        self.store.save(credential2, 'the only key')
        loaded = self.store.load('the only key')
        self.assertEqual(credential2.consumer.key, loaded.consumer.key)