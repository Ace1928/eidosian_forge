from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class DNSNameConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(DNSNameConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.constraint = cc.DNSNameConstraint()

    def test_validation(self):
        self.assertTrue(self.constraint.validate('openstack.org.', self.ctx))

    def test_validation_error_hyphen(self):
        dns_name = '-openstack.org'
        expected = "'%s' not in valid format. Reason: Name '%s' must not start or end with a hyphen." % (dns_name, dns_name.split('.')[0])
        self.assertFalse(self.constraint.validate(dns_name, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_error_empty_component(self):
        dns_name = '.openstack.org'
        expected = "'%s' not in valid format. Reason: Encountered an empty component." % dns_name
        self.assertFalse(self.constraint.validate(dns_name, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_error_special_char(self):
        dns_name = '$openstack.org'
        expected = "'%s' not in valid format. Reason: Name '%s' must be 1-63 characters long, each of which can only be alphanumeric or a hyphen." % (dns_name, dns_name.split('.')[0])
        self.assertFalse(self.constraint.validate(dns_name, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_error_tld_allnumeric(self):
        dns_name = 'openstack.123.'
        expected = "'%s' not in valid format. Reason: TLD '%s' must not be all numeric." % (dns_name, dns_name.split('.')[1])
        self.assertFalse(self.constraint.validate(dns_name, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_none(self):
        self.assertTrue(self.constraint.validate(None, self.ctx))