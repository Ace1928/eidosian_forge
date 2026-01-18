from unittest import mock
from heat.engine.constraint import common_constraints as cc
from heat.tests import common
from heat.tests import utils
def test_validation_out_of_range_error(self):
    cron_expression = '* * * * * 100'
    expect = 'Invalid CRON expression: [%s] is not acceptable, out of range' % cron_expression
    self.assertFalse(self.constraint.validate(cron_expression, self.ctx))
    self.assertEqual(expect, str(self.constraint._error_message))