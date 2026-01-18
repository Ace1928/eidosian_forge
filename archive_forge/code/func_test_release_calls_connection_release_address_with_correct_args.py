from tests.compat import mock, unittest
from boto.ec2.address import Address
def test_release_calls_connection_release_address_with_correct_args(self):
    self.address.release()
    self.address.connection.release_address.assert_called_with(allocation_id='aid1', dry_run=False)