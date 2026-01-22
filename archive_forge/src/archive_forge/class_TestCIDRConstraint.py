from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class TestCIDRConstraint(common.HeatTestCase):

    def setUp(self):
        super(TestCIDRConstraint, self).setUp()
        self.constraint = cc.CIDRConstraint()

    def test_valid_cidr_format(self):
        validate_format = ['10.0.0.0/24', '6000::/64']
        for cidr in validate_format:
            self.assertTrue(self.constraint.validate(cidr, None))

    def test_invalid_cidr_format(self):
        invalidate_format = ['::/129', 'Invalid cidr', '300.0.0.0/24', '10.0.0.0/33', '10.0.0/24', '10.0/24', '10.0.a.10/24', '8.8.8.0/ 24', '8.8.8.8']
        for cidr in invalidate_format:
            self.assertFalse(self.constraint.validate(cidr, None))

    @mock.patch('neutron_lib.api.validators.validate_subnet')
    def test_validate(self, mock_validate_subnet):
        test_formats = ['10.0.0/24', '10.0/24']
        self.assertFalse(self.constraint.validate('10.0.0.0/33', None))
        for cidr in test_formats:
            self.assertFalse(self.constraint.validate(cidr, None))
        mock_validate_subnet.assert_any_call('10.0.0/24')
        mock_validate_subnet.assert_called_with('10.0/24')
        self.assertEqual(mock_validate_subnet.call_count, 2)