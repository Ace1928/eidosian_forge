from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_update_with_validate_true_raises_value_error(self):
    self.eni_one.connection = mock.Mock()
    self.eni_one.connection.get_all_network_interfaces.return_value = []
    with self.assertRaisesRegexp(ValueError, '^eni-1 is not a valid ENI ID$'):
        self.eni_one.update(True)