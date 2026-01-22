import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestKeyringCredentialStore(CredentialStoreTestCase):
    """Tests for the KeyringCredentialStore class."""

    def setUp(self):
        self.keyring = InMemoryKeyring()
        self.store = KeyringCredentialStore()

    def test_save_and_load(self):
        with fake_keyring(self.keyring):
            credential = self.make_credential('consumer key')
            self.store.save(credential, 'unique key')
            credential2 = self.store.load('unique key')
            self.assertEqual(credential.consumer.key, credential2.consumer.key)

    def test_lookup_by_unique_key(self):
        with fake_keyring(self.keyring):
            credential1 = self.make_credential('consumer key1')
            self.store.save(credential1, 'key 1')
            credential2 = self.make_credential('consumer key2')
            self.store.save(credential2, 'key 2')
            loaded1 = self.store.load('key 1')
            self.assertTrue(loaded1)
            self.assertEqual(credential1.consumer.key, loaded1.consumer.key)
            loaded2 = self.store.load('key 2')
            self.assertEqual(credential2.consumer.key, loaded2.consumer.key)

    def test_reused_unique_id_overwrites_old_credential(self):
        with fake_keyring(self.keyring):
            credential1 = self.make_credential('consumer key1')
            self.store.save(credential1, 'the only key')
            credential2 = self.make_credential('consumer key2')
            self.store.save(credential2, 'the only key')
            loaded = self.store.load('the only key')
            self.assertEqual(credential2.consumer.key, loaded.consumer.key)

    def test_bad_unique_id_returns_none(self):
        with fake_keyring(self.keyring):
            self.assertIsNone(self.store.load('no such key'))

    def test_keyring_returns_unicode(self):

        class UnicodeInMemoryKeyring(InMemoryKeyring):

            def get_password(self, service, username):
                password = super(UnicodeInMemoryKeyring, self).get_password(service, username)
                if isinstance(password, unicode_type):
                    password = password.encode('utf-8')
                return password
        self.keyring = UnicodeInMemoryKeyring()
        with fake_keyring(self.keyring):
            credential = self.make_credential('consumer key')
            self.assertTrue(credential)
            self.store.save(credential, 'unique key')
            credential2 = self.store.load('unique key')
            self.assertTrue(credential2)
            self.assertEqual(credential.consumer.key, credential2.consumer.key)
            self.assertEqual(credential.consumer.secret, credential2.consumer.secret)

    def test_nonencoded_key_handled(self):

        class UnencodedInMemoryKeyring(InMemoryKeyring):

            def get_password(self, service, username):
                pw = super(UnencodedInMemoryKeyring, self).get_password(service, username)
                return b64decode(pw[5:])
        self.keyring = UnencodedInMemoryKeyring()
        with fake_keyring(self.keyring):
            credential = self.make_credential('consumer key')
            self.store.save(credential, 'unique key')
            credential2 = self.store.load('unique key')
            self.assertEqual(credential.consumer.key, credential2.consumer.key)
            self.assertEqual(credential.consumer.secret, credential2.consumer.secret)

    def test_corrupted_key_handled(self):

        class CorruptedInMemoryKeyring(InMemoryKeyring):

            def get_password(self, service, username):
                return 'bad'
        self.keyring = CorruptedInMemoryKeyring()
        with fake_keyring(self.keyring):
            credential = self.make_credential('consumer key')
            self.store.save(credential, 'unique key')
            credential2 = self.store.load('unique key')
            self.assertIsNone(credential2)