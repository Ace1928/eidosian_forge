import platform
import sys
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_service import loopingcall
from os_brick import exception
from os_brick.initiator import connector
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fake
from os_brick.initiator.connectors import iscsi
from os_brick.initiator.connectors import nvmeof
from os_brick.initiator import linuxfc
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick import utils
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_native_multipath_supported', return_value=False)
@mock.patch.object(priv_nvmeof, 'get_system_uuid', return_value=None)
@mock.patch.object(nvmeof.NVMeOFConnector, '_get_host_uuid', return_value=None)
@mock.patch.object(utils, 'get_host_nqn', return_value=None)
@mock.patch.object(iscsi.ISCSIConnector, 'get_initiator', return_value='fakeinitiator')
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_wwpns', return_value=None)
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_wwnns', return_value=None)
@mock.patch.object(platform, 'machine', mock.Mock(return_value='s390x'))
@mock.patch('sys.platform', 'linux2')
@mock.patch.object(utils, 'get_nvme_host_id', mock.Mock(return_value=None))
def _test_brick_get_connector_properties(self, multipath, enforce_multipath, multipath_result, mock_wwnns, mock_wwpns, mock_initiator, mock_nqn, mock_hostuuid, mock_sysuuid, mock_native_multipath_supported, host='fakehost'):
    props_actual = connector.get_connector_properties('sudo', MY_IP, multipath, enforce_multipath, host=host)
    os_type = 'linux2'
    platform = 's390x'
    props = {'initiator': 'fakeinitiator', 'host': host, 'ip': MY_IP, 'multipath': multipath_result, 'nvme_native_multipath': False, 'os_type': os_type, 'platform': platform, 'do_local_attach': False}
    self.assertEqual(props, props_actual)