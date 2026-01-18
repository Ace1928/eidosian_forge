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
@ddt.data(('share_instance_xyz',), ('share_instance_abc', 'share_instance_xyz'))
def test_share_instance_force_delete_wait(self, instances_to_delete):
    fake_manager = mock.Mock()
    fake_instances = [share_instances.ShareInstance(fake_manager, {'id': '1234'}) for instance in instances_to_delete]
    instance_not_found_error = "Delete for instance %s failed: No instance with a name or ID of '%s' exists."
    instances_are_not_found_errors = [exceptions.CommandError(instance_not_found_error % (instance, instance)) for instance in instances_to_delete]
    self.mock_object(shell_v2, '_find_share_instance', mock.Mock(side_effect=fake_instances + instances_are_not_found_errors))
    self.run_command('share-instance-force-delete %s --wait' % ' '.join(instances_to_delete))
    shell_v2._find_share_instance.assert_has_calls([mock.call(self.shell.cs, instance) for instance in instances_to_delete])
    fake_manager.force_delete.assert_has_calls([mock.call(instance) for instance in fake_instances])
    self.assertEqual(len(instances_to_delete), fake_manager.force_delete.call_count)