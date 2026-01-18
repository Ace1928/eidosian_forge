from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.assured import message_util
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.core import resources
Acknowledge an existing Assured Workloads compliance violation.

    Args:
      name: str, the name for the Assured Workloads violation being described in
        the form:
        organizations/{ORG_ID}/locations/{LOCATION}/workloads/{WORKLOAD_ID}/violations/{VIOLATION_ID}.
      comment: str, the business justification which the user wants to add while
        acknowledging a violation.
      acknowledge_type: str, the acknowledge type for specified violation, which
        is one of: SINGLE_VIOLATION - to acknowledge specified violation,
        EXISTING_CHILD_RESOURCE_VIOLATIONS - to acknowledge specified org policy
        violation and all associated child resource violations.

    Returns:
      Specified Assured Workloads Violation.
    