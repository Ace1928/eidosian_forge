from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_datetime_format_with_utc_offset(self):
    date = '2050-01-01T23:59:59+00:00'
    self.assertTrue(self.constraint.validate(date, None))