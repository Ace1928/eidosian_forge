import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_save_and_load(self):
    with fake_keyring(self.keyring):
        credential = self.make_credential('consumer key')
        self.store.save(credential, 'unique key')
        credential2 = self.store.load('unique key')
        self.assertEqual(credential.consumer.key, credential2.consumer.key)