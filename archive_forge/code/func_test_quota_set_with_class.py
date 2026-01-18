import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_quota_set_with_class(self):
    arglist = ['--injected-files', str(compute_fakes.injected_file_num), '--injected-file-size', str(compute_fakes.injected_file_size_num), '--injected-path-size', str(compute_fakes.injected_path_size_num), '--key-pairs', str(compute_fakes.key_pair_num), '--cores', str(compute_fakes.core_num), '--ram', str(compute_fakes.ram_num), '--instances', str(compute_fakes.instance_num), '--properties', str(compute_fakes.property_num), '--server-groups', str(compute_fakes.servgroup_num), '--server-group-members', str(compute_fakes.servgroup_members_num), '--gigabytes', str(compute_fakes.floating_ip_num), '--snapshots', str(compute_fakes.fix_ip_num), '--volumes', str(volume_fakes.QUOTA['volumes']), '--network', str(network_fakes.QUOTA['network']), '--class', self.projects[0].name]
    verifylist = [('injected_files', compute_fakes.injected_file_num), ('injected_file_content_bytes', compute_fakes.injected_file_size_num), ('injected_file_path_bytes', compute_fakes.injected_path_size_num), ('key_pairs', compute_fakes.key_pair_num), ('cores', compute_fakes.core_num), ('ram', compute_fakes.ram_num), ('instances', compute_fakes.instance_num), ('metadata_items', compute_fakes.property_num), ('server_groups', compute_fakes.servgroup_num), ('server_group_members', compute_fakes.servgroup_members_num), ('gigabytes', compute_fakes.floating_ip_num), ('snapshots', compute_fakes.fix_ip_num), ('volumes', volume_fakes.QUOTA['volumes']), ('network', network_fakes.QUOTA['network']), ('quota_class', True), ('project', self.projects[0].name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs_compute = {'injected_files': compute_fakes.injected_file_num, 'injected_file_content_bytes': compute_fakes.injected_file_size_num, 'injected_file_path_bytes': compute_fakes.injected_path_size_num, 'key_pairs': compute_fakes.key_pair_num, 'cores': compute_fakes.core_num, 'ram': compute_fakes.ram_num, 'instances': compute_fakes.instance_num, 'metadata_items': compute_fakes.property_num, 'server_groups': compute_fakes.servgroup_num, 'server_group_members': compute_fakes.servgroup_members_num}
    kwargs_volume = {'gigabytes': compute_fakes.floating_ip_num, 'snapshots': compute_fakes.fix_ip_num, 'volumes': volume_fakes.QUOTA['volumes']}
    self.compute_quotas_class_mock.update.assert_called_with(self.projects[0].name, **kwargs_compute)
    self.volume_quotas_class_mock.update.assert_called_with(self.projects[0].name, **kwargs_volume)
    self.assertNotCalled(self.network_client.update_quota)
    self.assertIsNone(result)