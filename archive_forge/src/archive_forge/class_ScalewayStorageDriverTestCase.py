import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.s3 import S3SignatureV4Connection
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.storage.drivers.scaleway import SCW_FR_PAR_STANDARD_HOST, ScalewayStorageDriver
class ScalewayStorageDriverTestCase(S3Tests, unittest.TestCase):
    driver_type = ScalewayStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = SCW_FR_PAR_STANDARD_HOST

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, host=self.default_host)

    def setUp(self):
        super().setUp()
        ScalewayStorageDriver.connectionCls.conn_class = S3MockHttp
        S3MockHttp.type = None
        self.driver = self.create_driver()

    def test_connection_class_type(self):
        self.assertEqual(self.driver.connectionCls, S3SignatureV4Connection)

    def test_connection_class_default_host(self):
        self.assertEqual(self.driver.connectionCls.host, self.default_host)
        self.assertEqual(self.driver.connectionCls.port, 443)
        self.assertEqual(self.driver.connectionCls.secure, True)