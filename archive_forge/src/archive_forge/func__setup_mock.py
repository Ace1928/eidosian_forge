from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def _setup_mock(self):
    """Sets up a mock elb request.
        Returns: response, elb connection and LoadBalancer
        """
    mock_response = mock.Mock()
    mock_response.status = 200
    elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
    elb.make_request = mock.Mock(return_value=mock_response)
    return (mock_response, elb, LoadBalancer(elb, 'test_elb'))