from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_columns_length_error(self):
    cron_expression = '* *'
    expect = 'Invalid CRON expression: Exactly 5 or 6 columns has to be specified for '
    self.assertFalse(self.constraint.validate(cron_expression, self.ctx))
    self.assertIn(expect, str(self.constraint._error_message))