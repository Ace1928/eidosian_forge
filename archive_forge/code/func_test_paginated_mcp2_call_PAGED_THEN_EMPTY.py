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
def test_paginated_mcp2_call_PAGED_THEN_EMPTY(self):
    self.driver.connection._get_orgId()
    DimensionDataMockHttp.type = 'PAGED_THEN_EMPTY'
    node_list_generator = self.driver.connection.paginated_request_with_orgId_api_2('server/server')
    final_node_list = []
    for node_list in node_list_generator:
        final_node_list.extend(node_list)
    self.assertTrue(len(final_node_list) == 2)