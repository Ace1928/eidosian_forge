from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
class TestISO8601Constraint(common.HeatTestCase):

    def setUp(self):
        super(TestISO8601Constraint, self).setUp()
        self.constraint = cc.ISO8601Constraint()

    def test_validate_date_format(self):
        date = '2050-01-01'
        self.assertTrue(self.constraint.validate(date, None))

    def test_validate_datetime_format(self):
        self.assertTrue(self.constraint.validate('2050-01-01T23:59:59', None))

    def test_validate_datetime_format_with_utc_offset(self):
        date = '2050-01-01T23:59:59+00:00'
        self.assertTrue(self.constraint.validate(date, None))

    def test_validate_datetime_format_with_utc_offset_alternate(self):
        date = '2050-01-01T23:59:59+0000'
        self.assertTrue(self.constraint.validate(date, None))

    def test_validate_refuses_other_formats(self):
        self.assertFalse(self.constraint.validate('Fri 13th, 2050', None))