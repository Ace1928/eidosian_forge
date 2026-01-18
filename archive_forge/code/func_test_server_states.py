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
def test_server_states(self):
    DimensionDataMockHttp.type = None
    ret = self.driver.list_nodes()
    self.assertTrue(ret[0].state == 'running')
    self.assertTrue(ret[1].state == 'starting')
    self.assertTrue(ret[2].state == 'stopping')
    self.assertTrue(ret[3].state == 'reconfiguring')
    self.assertTrue(ret[4].state == 'running')
    self.assertTrue(ret[5].state == 'terminated')
    self.assertTrue(ret[6].state == 'stopped')
    self.assertEqual(len(ret), 7)