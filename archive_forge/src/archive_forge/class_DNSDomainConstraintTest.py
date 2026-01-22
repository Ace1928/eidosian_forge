from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class DNSDomainConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(DNSDomainConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.constraint = cc.DNSDomainConstraint()

    def test_validation(self):
        self.assertTrue(self.constraint.validate('openstack.org.', self.ctx))

    def test_validation_error_no_end_period(self):
        dns_domain = 'openstack.org'
        expected = "'%s' must end with '.'." % dns_domain
        self.assertFalse(self.constraint.validate(dns_domain, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_none(self):
        self.assertTrue(self.constraint.validate(None, self.ctx))