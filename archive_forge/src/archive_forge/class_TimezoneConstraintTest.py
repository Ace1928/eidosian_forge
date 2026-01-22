from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class TimezoneConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(TimezoneConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.constraint = cc.TimezoneConstraint()

    def test_validation(self):
        self.assertTrue(self.constraint.validate('Asia/Taipei', self.ctx))

    def test_validation_error(self):
        timezone = 'wrong_timezone'
        err = timezone
        if zoneinfo:
            err = 'No time zone found with key %s' % timezone
        expected = "Invalid timezone: '%s'" % err
        self.assertFalse(self.constraint.validate(timezone, self.ctx))
        self.assertEqual(expected, str(self.constraint._error_message))

    def test_validation_none(self):
        self.assertTrue(self.constraint.validate(None, self.ctx))