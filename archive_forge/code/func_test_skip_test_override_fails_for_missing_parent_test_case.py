import sys
import warnings
from oslo_log import log
from sqlalchemy import exc
from testtools import matchers
from keystone.tests import unit
def test_skip_test_override_fails_for_missing_parent_test_case(self):
    test = self.TestChild('test_not_in_parent')
    result = test.run()
    observed_error = result.decorated.errors[0]
    observed_error_msg = observed_error[1]
    expected_error_msg = "'test_not_in_parent' is not a previously defined test method"
    self.assertIn(expected_error_msg, observed_error_msg)