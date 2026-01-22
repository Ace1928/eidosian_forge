from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
Map a test roll-up outcome into the appropriate gcloud test exit_code.

  Args:
    outcome: a toolresults_v1.Outcome message.
    summary_enum: a toolresults.Outcome.SummaryValueValuesEnum reference.

  Returns:
    The exit_code which corresponds to the test execution's rolled-up outcome.

  Raises:
    TestOutcomeError: If Tool Results service returns an invalid outcome value.
  