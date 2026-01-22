import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
class TestFernetKeyRotation(unit.TestCase):

    def setUp(self):
        super(TestFernetKeyRotation, self).setUp()
        self.key_repo_signatures = set()

    @property
    def keys(self):
        """Key files converted to numbers."""
        return sorted((int(x) for x in os.listdir(CONF.fernet_receipts.key_repository)))

    @property
    def key_repository_size(self):
        """The number of keys in the key repository."""
        return len(self.keys)

    @property
    def key_repository_signature(self):
        """Create a "thumbprint" of the current key repository.

        Because key files are renamed, this produces a hash of the contents of
        the key files, ignoring their filenames.

        The resulting signature can be used, for example, to ensure that you
        have a unique set of keys after you perform a key rotation (taking a
        static set of keys, and simply shuffling them, would fail such a test).

        """
        key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
        keys = key_utils.load_keys()
        keys.sort()
        signature = hashlib.sha1()
        for key in keys:
            signature.update(key.encode('utf-8'))
        return signature.hexdigest()

    def assertRepositoryState(self, expected_size):
        """Validate the state of the key repository."""
        self.assertEqual(expected_size, self.key_repository_size)
        self.assertUniqueRepositoryState()

    def assertUniqueRepositoryState(self):
        """Ensure that the current key repo state has not been seen before."""
        signature = self.key_repository_signature
        self.assertNotIn(signature, self.key_repo_signatures)
        self.key_repo_signatures.add(signature)

    def test_rotation(self):
        min_active_keys = 2
        for max_active_keys in range(min_active_keys, 52 + 1):
            self.config_fixture.config(group='fernet_receipts', max_active_keys=max_active_keys)
            self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_receipts', CONF.fernet_receipts.max_active_keys))
            self.assertRepositoryState(expected_size=min_active_keys)
            exp_keys = [0, 1]
            next_key_number = exp_keys[-1] + 1
            self.assertEqual(exp_keys, self.keys)
            key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
            for rotation in range(max_active_keys - min_active_keys):
                key_utils.rotate_keys()
                self.assertRepositoryState(expected_size=rotation + 3)
                exp_keys.append(next_key_number)
                next_key_number += 1
                self.assertEqual(exp_keys, self.keys)
            self.assertEqual(max_active_keys, self.key_repository_size)
            key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
            for rotation in range(10):
                key_utils.rotate_keys()
                self.assertRepositoryState(expected_size=max_active_keys)
                exp_keys.pop(1)
                exp_keys.append(next_key_number)
                next_key_number += 1
                self.assertEqual(exp_keys, self.keys)

    def test_rotation_disk_write_fail(self):
        self.assertRepositoryState(expected_size=2)
        key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
        mock_open = mock.mock_open()
        file_handle = mock_open()
        file_handle.flush.side_effect = IOError('disk full')
        with mock.patch('keystone.common.fernet_utils.open', mock_open):
            self.assertRaises(IOError, key_utils.rotate_keys)
        self.assertEqual(self.key_repository_size, 2)
        with mock.patch('keystone.common.fernet_utils.open', mock_open):
            self.assertRaises(IOError, key_utils.rotate_keys)
        self.assertEqual(self.key_repository_size, 2)
        key_utils.rotate_keys()
        self.assertEqual(self.key_repository_size, 3)

    def test_rotation_empty_file(self):
        active_keys = 2
        self.assertRepositoryState(expected_size=active_keys)
        empty_file = os.path.join(CONF.fernet_receipts.key_repository, '2')
        with open(empty_file, 'w'):
            pass
        key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
        key_utils.rotate_keys()
        self.assertTrue(os.path.isfile(empty_file))
        keys = key_utils.load_keys()
        self.assertEqual(3, len(keys))
        self.assertTrue(os.path.getsize(empty_file) > 0)

    def test_non_numeric_files(self):
        evil_file = os.path.join(CONF.fernet_receipts.key_repository, '99.bak')
        with open(evil_file, 'w'):
            pass
        key_utils = fernet_utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
        key_utils.rotate_keys()
        self.assertTrue(os.path.isfile(evil_file))
        keys = 0
        for x in os.listdir(CONF.fernet_receipts.key_repository):
            if x == '99.bak':
                continue
            keys += 1
        self.assertEqual(3, keys)