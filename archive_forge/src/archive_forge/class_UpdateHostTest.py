import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
class UpdateHostTest(tests.TestCase):

    def create_update_command(self, list_value):
        mock_host_manager = mock.Mock()
        mock_host_manager.list.return_value = list_value
        mock_client = mock.Mock()
        mock_client.host = mock_host_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (hosts.UpdateHost(blazar_shell, mock.Mock()), mock_host_manager)

    def test_update_host(self):
        list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
        update_host, host_manager = self.create_update_command(list_value)
        args = argparse.Namespace(id='101', extra_capabilities=['key1=value1', 'key2=value2'])
        expected = {'values': {'key1': 'value1', 'key2': 'value2'}}
        update_host.run(args)
        host_manager.update.assert_called_once_with('101', **expected)

    def test_update_host_with_name(self):
        list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
        update_host, host_manager = self.create_update_command(list_value)
        args = argparse.Namespace(id='host-1', extra_capabilities=['key1=value1', 'key2=value2'])
        expected = {'values': {'key1': 'value1', 'key2': 'value2'}}
        update_host.run(args)
        host_manager.update.assert_called_once_with('101', **expected)

    def test_update_host_with_name_startwith_number(self):
        list_value = [{'id': '101', 'hypervisor_hostname': '1-host'}, {'id': '201', 'hypervisor_hostname': '2-host'}]
        update_host, host_manager = self.create_update_command(list_value)
        args = argparse.Namespace(id='1-host', extra_capabilities=['key1=value1', 'key2=value2'])
        expected = {'values': {'key1': 'value1', 'key2': 'value2'}}
        update_host.run(args)
        host_manager.update.assert_called_once_with('101', **expected)