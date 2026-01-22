from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class ExpirationConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(ExpirationConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.constraint = cc.ExpirationConstraint()

    def test_validate_date_format(self):
        date = '2050-01-01'
        self.assertTrue(self.constraint.validate(date, None))

    def test_validation_error(self):
        expiration = 'Fri 13th, 2050'
        expected = "Expiration {0} is invalid: Unable to parse date string '{0}'".format(expiration)
        self.assertFalse(self.constraint.validate(expiration, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_before_current_time(self):
        expiration = '1970-01-01'
        expected = 'Expiration %s is invalid: Expiration time is out of date.' % expiration
        self.assertFalse(self.constraint.validate(expiration, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_none(self):
        self.assertTrue(self.constraint.validate(None, self.ctx))