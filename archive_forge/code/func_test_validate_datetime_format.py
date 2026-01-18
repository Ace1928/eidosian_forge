from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validate_datetime_format(self):
    self.assertTrue(self.constraint.validate('2050-01-01T23:59:59', None))