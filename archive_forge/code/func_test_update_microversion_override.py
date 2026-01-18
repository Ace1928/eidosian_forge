import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_update_microversion_override(self):
    patch = {'op': 'replace', 'value': NEW_DRIVER, 'path': '/driver'}
    node = self.mgr.update(node_id=NODE1['uuid'], patch=patch, os_ironic_api_version='1.24')
    expect = [('PATCH', '/v1/nodes/%s' % NODE1['uuid'], {'X-OpenStack-Ironic-API-Version': '1.24'}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_DRIVER, node.driver)