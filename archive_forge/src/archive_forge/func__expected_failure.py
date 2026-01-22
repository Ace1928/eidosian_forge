from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def _expected_failure(case):
    case.expectFailure('arbitrary expected failure', _failure, case)