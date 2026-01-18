import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_show_lease(self):
    show_lease, lease_manager = self.create_show_command()
    lease_manager.get.return_value = {'id': FIRST_LEASE}
    args = argparse.Namespace(id=FIRST_LEASE)
    expected = [('id',), (FIRST_LEASE,)]
    self.assertEqual(show_lease.get_data(args), expected)
    lease_manager.get.assert_called_once_with(FIRST_LEASE)