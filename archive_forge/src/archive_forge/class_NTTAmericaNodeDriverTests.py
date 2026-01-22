import unittest
from libcloud.compute.drivers.ntta import NTTAmericaNodeDriver
from libcloud.test.compute.test_dimensiondata_v2_3 import (
class NTTAmericaNodeDriverTests(DimensionData_v2_3_Tests, unittest.TestCase):

    def setUp(self):
        NTTAmericaNodeDriver.connectionCls.conn_class = DimensionDataMockHttp
        NTTAmericaNodeDriver.connectionCls.active_api_version = '2.3'
        DimensionDataMockHttp.type = None
        self.driver = NTTAmericaNodeDriver('user', 'password')