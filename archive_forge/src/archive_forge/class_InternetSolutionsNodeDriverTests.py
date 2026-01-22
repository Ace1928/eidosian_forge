import unittest
from libcloud.compute.drivers.internetsolutions import InternetSolutionsNodeDriver
from libcloud.test.compute.test_dimensiondata_v2_3 import (
class InternetSolutionsNodeDriverTests(DimensionData_v2_3_Tests, unittest.TestCase):

    def setUp(self):
        InternetSolutionsNodeDriver.connectionCls.conn_class = DimensionDataMockHttp
        InternetSolutionsNodeDriver.connectionCls.active_api_version = '2.3'
        DimensionDataMockHttp.type = None
        self.driver = InternetSolutionsNodeDriver('user', 'password')