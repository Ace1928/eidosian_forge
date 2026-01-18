from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_date_format(self):
    date = '2050-01-01'
    self.assertTrue(self.constraint.validate(date, None))