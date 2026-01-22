import boto
from boto.kms.exceptions import NotFoundException
from tests.compat import unittest
class TestKMS(unittest.TestCase):

    def setUp(self):
        self.kms = boto.connect_kms()

    def test_list_keys(self):
        response = self.kms.list_keys()
        self.assertIn('Keys', response)

    def test_handle_not_found_exception(self):
        with self.assertRaises(NotFoundException):
            self.kms.describe_key(key_id='nonexistant_key')