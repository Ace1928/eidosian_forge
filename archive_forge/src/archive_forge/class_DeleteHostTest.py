import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import hosts
class DeleteHostTest(tests.TestCase):

    def create_delete_command(self, list_value):
        mock_host_manager = mock.Mock()
        mock_host_manager.list.return_value = list_value
        mock_client = mock.Mock()
        mock_client.host = mock_host_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (hosts.DeleteHost(blazar_shell, mock.Mock()), mock_host_manager)

    def test_delete_host(self):
        list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
        delete_host, host_manager = self.create_delete_command(list_value)
        args = argparse.Namespace(id='101')
        delete_host.run(args)
        host_manager.delete.assert_called_once_with('101')

    def test_delete_host_with_name(self):
        list_value = [{'id': '101', 'hypervisor_hostname': 'host-1'}, {'id': '201', 'hypervisor_hostname': 'host-2'}]
        delete_host, host_manager = self.create_delete_command(list_value)
        args = argparse.Namespace(id='host-1')
        delete_host.run(args)
        host_manager.delete.assert_called_once_with('101')

    def test_delete_host_with_name_startwith_number(self):
        list_value = [{'id': '101', 'hypervisor_hostname': '1-host'}, {'id': '201', 'hypervisor_hostname': '2-host'}]
        delete_host, host_manager = self.create_delete_command(list_value)
        args = argparse.Namespace(id='1-host')
        delete_host.run(args)
        host_manager.delete.assert_called_once_with('101')