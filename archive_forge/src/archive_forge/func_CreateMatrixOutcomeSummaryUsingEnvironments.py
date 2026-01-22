from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def CreateMatrixOutcomeSummaryUsingEnvironments(self):
    """Fetches test results and creates a test outcome summary.

    Lists all the environments in an execution and creates a high-level outcome
    summary for each environment (pass/flaky/fail/skipped/inconclusive). Each
    environment represents a combination of one or more test executions with the
    same device configuration (e.g. running the tests on a Nexus 5 in portrait
    mode using the en locale and API level 18).

    Returns:
      A list of TestOutcome objects.

    Raises:
      HttpException if the ToolResults service reports a back-end error.
    """
    outcomes = []
    environments = self._ListAllEnvironments()
    if not environments:
        log.warning('Environment has no results, something went wrong. Displaying step outcomes instead.')
        return self.CreateMatrixOutcomeSummaryUsingSteps()
    for environment in environments:
        dimension_value = environment.dimensionValue
        axis_value = self._GetAxisValue(dimension_value)
        if not environment.environmentResult.outcome:
            log.warning('Environment for [{0}] had no outcome value. Displaying step outcomes instead.'.format(axis_value))
            return self.CreateMatrixOutcomeSummaryUsingSteps()
        details = self._GetEnvironmentOutcomeDetails(environment)
        self._LogWarnings(details, axis_value)
        outcome_summary = environment.environmentResult.outcome.summary
        outcome_str = self._GetOutcomeSummaryDisplayName(outcome_summary)
        outcomes.append(TestOutcome(outcome=outcome_str, axis_value=axis_value, test_details=details))
    return sorted(outcomes, key=_TestOutcomeSortKey)