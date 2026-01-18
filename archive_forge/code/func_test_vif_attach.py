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
@mock.patch.object(node.NodeManager, 'update', autospec=True)
def test_vif_attach(self, update_mock):
    kwargs = {'node_ident': NODE1['uuid'], 'vif_id': 'vif_id'}
    final_path = '%s/vifs' % NODE1['uuid']
    self.mgr.vif_attach(**kwargs)
    update_mock.assert_called_once_with(mock.ANY, final_path, {'id': 'vif_id'}, http_method='POST', os_ironic_api_version=None, global_request_id=None)