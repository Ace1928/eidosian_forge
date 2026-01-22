import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
class ShowLeaseTestCase(tests.TestCase):

    def create_show_command(self):
        mock_lease_manager = mock.Mock()
        mock_client = mock.Mock()
        mock_client.lease = mock_lease_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (leases.ShowLease(blazar_shell, mock.Mock()), mock_lease_manager)

    def test_show_lease(self):
        show_lease, lease_manager = self.create_show_command()
        lease_manager.get.return_value = {'id': FIRST_LEASE}
        args = argparse.Namespace(id=FIRST_LEASE)
        expected = [('id',), (FIRST_LEASE,)]
        self.assertEqual(show_lease.get_data(args), expected)
        lease_manager.get.assert_called_once_with(FIRST_LEASE)

    def test_show_lease_by_name(self):
        show_lease, lease_manager = self.create_show_command()
        lease_manager.list.return_value = [{'id': FIRST_LEASE, 'name': 'first-lease'}, {'id': SECOND_LEASE, 'name': 'second-lease'}]
        lease_manager.get.return_value = {'id': SECOND_LEASE}
        args = argparse.Namespace(id='second-lease')
        expected = [('id',), (SECOND_LEASE,)]
        self.assertEqual(show_lease.get_data(args), expected)
        lease_manager.list.assert_called_once_with()
        lease_manager.get.assert_called_once_with(SECOND_LEASE)