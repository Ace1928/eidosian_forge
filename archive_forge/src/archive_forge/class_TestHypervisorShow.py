import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
class TestHypervisorShow(compute_fakes.TestComputev2):

    def setUp(self):
        super().setUp()
        uptime_string = ' 01:28:24 up 3 days, 11:15,  1 user,  load average: 0.94, 0.62, 0.50\n'
        self.hypervisor = compute_fakes.create_one_hypervisor(attrs={'uptime': uptime_string})
        self.compute_sdk_client.find_hypervisor.return_value = self.hypervisor
        self.compute_sdk_client.aggregates.return_value = []
        uptime_info = {'status': self.hypervisor.status, 'state': self.hypervisor.state, 'id': self.hypervisor.id, 'hypervisor_hostname': self.hypervisor.name, 'uptime': uptime_string}
        self.compute_sdk_client.get_hypervisor_uptime.return_value = uptime_info
        self.columns_v288 = ('aggregates', 'cpu_info', 'host_ip', 'host_time', 'hypervisor_hostname', 'hypervisor_type', 'hypervisor_version', 'id', 'load_average', 'service_host', 'service_id', 'state', 'status', 'uptime', 'users')
        self.data_v288 = ([], format_columns.DictColumn({'aaa': 'aaa'}), '192.168.0.10', '01:28:24', self.hypervisor.name, 'QEMU', 2004001, self.hypervisor.id, '0.94, 0.62, 0.50', 'aaa', 1, 'up', 'enabled', '3 days, 11:15', '1')
        self.columns = ('aggregates', 'cpu_info', 'current_workload', 'disk_available_least', 'free_disk_gb', 'free_ram_mb', 'host_ip', 'host_time', 'hypervisor_hostname', 'hypervisor_type', 'hypervisor_version', 'id', 'load_average', 'local_gb', 'local_gb_used', 'memory_mb', 'memory_mb_used', 'running_vms', 'service_host', 'service_id', 'state', 'status', 'uptime', 'users', 'vcpus', 'vcpus_used')
        self.data = ([], format_columns.DictColumn({'aaa': 'aaa'}), 0, 50, 50, 1024, '192.168.0.10', '01:28:24', self.hypervisor.name, 'QEMU', 2004001, self.hypervisor.id, '0.94, 0.62, 0.50', 50, 0, 1024, 512, 0, 'aaa', 1, 'up', 'enabled', '3 days, 11:15', '1', 4, 0)
        self.cmd = hypervisor.ShowHypervisor(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_hypervisor_show(self, sm_mock):
        arglist = [self.hypervisor.name]
        verifylist = [('hypervisor', self.hypervisor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns_v288, columns)
        self.assertCountEqual(self.data_v288, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', side_effect=[False, True, False])
    def test_hypervisor_show_pre_v288(self, sm_mock):
        arglist = [self.hypervisor.name]
        verifylist = [('hypervisor', self.hypervisor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_hypervisor_show_pre_v228(self, sm_mock):
        self.hypervisor.cpu_info = json.dumps(self.hypervisor.cpu_info)
        self.compute_sdk_client.find_hypervisor.return_value = self.hypervisor
        arglist = [self.hypervisor.name]
        verifylist = [('hypervisor', self.hypervisor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch.object(sdk_utils, 'supports_microversion', side_effect=[False, True, False])
    def test_hypervisor_show_uptime_not_implemented(self, sm_mock):
        arglist = [self.hypervisor.name]
        verifylist = [('hypervisor', self.hypervisor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.get_hypervisor_uptime.side_effect = nova_exceptions.HTTPNotImplemented(501)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ('aggregates', 'cpu_info', 'current_workload', 'disk_available_least', 'free_disk_gb', 'free_ram_mb', 'host_ip', 'hypervisor_hostname', 'hypervisor_type', 'hypervisor_version', 'id', 'local_gb', 'local_gb_used', 'memory_mb', 'memory_mb_used', 'running_vms', 'service_host', 'service_id', 'state', 'status', 'vcpus', 'vcpus_used')
        expected_data = ([], format_columns.DictColumn({'aaa': 'aaa'}), 0, 50, 50, 1024, '192.168.0.10', self.hypervisor.name, 'QEMU', 2004001, self.hypervisor.id, 50, 0, 1024, 512, 0, 'aaa', 1, 'up', 'enabled', 4, 0)
        self.assertEqual(expected_columns, columns)
        self.assertCountEqual(expected_data, data)