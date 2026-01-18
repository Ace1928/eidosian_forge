import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def test_run_node_invalid_state(self):
    """
        Test 'ex_run_node' invalid state.

        Test the Driver raises an exception when try to run a
        node that is in invalid state to run.
        """
    self.driver = AbiquoNodeDriver('go', 'trunks', 'http://dummy.host.com/api')
    node = self.driver.list_nodes()[0]
    self.assertRaises(LibcloudError, self.driver.ex_run_node, node)