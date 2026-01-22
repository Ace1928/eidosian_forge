from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
@retry.RetryOnException(max_retrials=10, max_wait_ms=_MAX_WAIT_TIME_IN_MS, exponential_sleep_multiplier=1.6, sleep_ms=100, should_retry_if=_ShouldRetryHttpError)
def _AddIamPolicyBindingsWithRetry(self, project_ref, member_roles):
    """Adds iam bindings to project_ref's iam policy, with retry.

    Args:
      project_ref: The project for the binding
      member_roles: List of 2-tuples of the form [(member, role), ...].

    Returns:
      The updated IAM Policy
    """
    return projects_api.AddIamPolicyBindings(project_ref, member_roles)