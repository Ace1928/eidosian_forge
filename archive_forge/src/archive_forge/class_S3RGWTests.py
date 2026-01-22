import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.rgw import (
class S3RGWTests(unittest.TestCase):
    driver_type = S3RGWStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = 'localhost'

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, signature_version='2', host=self.default_host)

    def setUp(self):
        self.driver = self.create_driver()

    def test_connection_class_type(self):
        res = self.driver.connectionCls is S3RGWConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver.connectionCls.host
        self.assertEqual(host, self.default_host)