from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
class TestLbAttributes(unittest.TestCase):
    """Tests LB Attributes."""

    def _setup_mock(self):
        """Sets up a mock elb request.
        Returns: response, elb connection and LoadBalancer
        """
        mock_response = mock.Mock()
        mock_response.status = 200
        elb = ELBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        elb.make_request = mock.Mock(return_value=mock_response)
        return (mock_response, elb, LoadBalancer(elb, 'test_elb'))

    def _verify_attributes(self, attributes, attr_tests):
        """Verifies an LbAttributes object."""
        for attr, result in attr_tests:
            attr_result = attributes
            for sub_attr in attr.split('.'):
                attr_result = getattr(attr_result, sub_attr, None)
            self.assertEqual(attr_result, result)

    def test_get_all_lb_attributes(self):
        """Tests getting the LbAttributes from the elb.connection."""
        mock_response, elb, _ = self._setup_mock()
        for response, attr_tests in ATTRIBUTE_TESTS:
            mock_response.read.return_value = response
            attributes = elb.get_all_lb_attributes('test_elb')
            self.assertTrue(isinstance(attributes, LbAttributes))
            self._verify_attributes(attributes, attr_tests)

    def test_get_lb_attribute(self):
        """Tests getting a single attribute from elb.connection."""
        mock_response, elb, _ = self._setup_mock()
        tests = [('crossZoneLoadBalancing', True, ATTRIBUTE_GET_TRUE_CZL_RESPONSE), ('crossZoneLoadBalancing', False, ATTRIBUTE_GET_FALSE_CZL_RESPONSE)]
        for attr, value, response in tests:
            mock_response.read.return_value = response
            status = elb.get_lb_attribute('test_elb', attr)
            self.assertEqual(status, value)

    def test_modify_lb_attribute(self):
        """Tests setting the attributes from elb.connection."""
        mock_response, elb, _ = self._setup_mock()
        tests = [('crossZoneLoadBalancing', True, ATTRIBUTE_SET_CZL_TRUE_REQUEST), ('crossZoneLoadBalancing', False, ATTRIBUTE_SET_CZL_FALSE_REQUEST)]
        for attr, value, args in tests:
            mock_response.read.return_value = ATTRIBUTE_SET_RESPONSE
            result = elb.modify_lb_attribute('test_elb', attr, value)
            self.assertTrue(result)
            elb.make_request.assert_called_with(*args)

    def test_lb_get_attributes(self):
        """Tests the LbAttributes from the ELB object."""
        mock_response, _, lb = self._setup_mock()
        for response, attr_tests in ATTRIBUTE_TESTS:
            mock_response.read.return_value = response
            attributes = lb.get_attributes(force=True)
            self.assertTrue(isinstance(attributes, LbAttributes))
            self._verify_attributes(attributes, attr_tests)

    def test_lb_is_cross_zone_load_balancing(self):
        """Tests checking is_cross_zone_load_balancing."""
        mock_response, _, lb = self._setup_mock()
        tests = [(lb.is_cross_zone_load_balancing, [], True, ATTRIBUTE_GET_TRUE_CZL_RESPONSE), (lb.is_cross_zone_load_balancing, [], True, ATTRIBUTE_GET_FALSE_CZL_RESPONSE), (lb.is_cross_zone_load_balancing, [True], False, ATTRIBUTE_GET_FALSE_CZL_RESPONSE)]
        for method, args, result, response in tests:
            mock_response.read.return_value = response
            self.assertEqual(method(*args), result)

    def test_lb_enable_cross_zone_load_balancing(self):
        """Tests enabling cross zone balancing from LoadBalancer."""
        mock_response, elb, lb = self._setup_mock()
        mock_response.read.return_value = ATTRIBUTE_SET_RESPONSE
        self.assertTrue(lb.enable_cross_zone_load_balancing())
        elb.make_request.assert_called_with(*ATTRIBUTE_SET_CZL_TRUE_REQUEST)

    def test_lb_disable_cross_zone_load_balancing(self):
        """Tests disabling cross zone balancing from LoadBalancer."""
        mock_response, elb, lb = self._setup_mock()
        mock_response.read.return_value = ATTRIBUTE_SET_RESPONSE
        self.assertTrue(lb.disable_cross_zone_load_balancing())
        elb.make_request.assert_called_with(*ATTRIBUTE_SET_CZL_FALSE_REQUEST)

    def test_lb_get_connection_settings(self):
        """Tests checking connectionSettings attribute"""
        mock_response, elb, _ = self._setup_mock()
        attrs = [('idle_timeout', 30)]
        mock_response.read.return_value = ATTRIBUTE_GET_CS_RESPONSE
        attributes = elb.get_all_lb_attributes('test_elb')
        self.assertTrue(isinstance(attributes, LbAttributes))
        for attr, value in attrs:
            self.assertEqual(getattr(attributes.connecting_settings, attr), value)