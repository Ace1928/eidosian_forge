import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.rgw import (
class S3RGWOutscaleTests(S3RGWTests):
    driver_type = S3RGWOutscaleStorageDriver
    default_host = 'osu.eu-west-2.outscale.com'

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, signature_version='4')

    def test_connection_class_type(self):
        res = self.driver.connectionCls is S3RGWConnectionAWS4
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver.connectionCls.host
        self.assertEqual(host, self.default_host)