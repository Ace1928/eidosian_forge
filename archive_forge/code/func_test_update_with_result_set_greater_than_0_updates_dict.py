from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
def test_update_with_result_set_greater_than_0_updates_dict(self):
    self.eni_two.connection.get_all_network_interfaces.return_value = [self.eni_one]
    self.eni_two.update()
    assert all([self.eni_two.status == 'one_status', self.eni_two.id == 'eni-1', self.eni_two.attachment == self.attachment])