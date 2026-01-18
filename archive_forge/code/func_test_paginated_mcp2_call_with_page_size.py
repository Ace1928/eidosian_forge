import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_paginated_mcp2_call_with_page_size(self):
    self.driver.connection._get_orgId()
    DimensionDataMockHttp.type = 'PAGESIZE50'
    node_list_generator = self.driver.connection.paginated_request_with_orgId_api_2('server/server', page_size=50)
    self.assertTrue(isinstance(node_list_generator, GeneratorType))