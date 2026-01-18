import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
@ddt.data(('fake_security_service1',), ('fake_security_service1', 'fake_security_service2'))
def test_security_service_delete(self, ss_ids):
    fake_security_services = [security_services.SecurityService('fake', {'id': ss_id}, True) for ss_id in ss_ids]
    self.mock_object(shell_v2, '_find_security_service', mock.Mock(side_effect=fake_security_services))
    self.run_command('security-service-delete %s' % ' '.join(ss_ids))
    shell_v2._find_security_service.assert_has_calls([mock.call(self.shell.cs, ss_id) for ss_id in ss_ids])
    for ss in fake_security_services:
        self.assert_called_anytime('DELETE', '/security-services/%s' % ss.id, clear_callstack=False)