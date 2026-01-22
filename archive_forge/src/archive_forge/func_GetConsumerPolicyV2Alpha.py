import collections
import copy
import enum
import sys
from typing import List
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import http_retry
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
def GetConsumerPolicyV2Alpha(policy_name):
    """Make API call to get a consumer policy.

  Args:
    policy_name: The name of a consumer policy. Currently supported format
      '{resource_type}/{resource_name}/consumerPolicies/default'. For example,
      'projects/100/consumerPolicies/default'.

  Raises:
    exceptions.GetConsumerPolicyPermissionDeniedException: when getting a
      consumer policy fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The consumer policy
  """
    client = _GetClientInstance('v2alpha')
    messages = client.MESSAGES_MODULE
    request = messages.ServiceusageConsumerPoliciesGetRequest(name=policy_name)
    try:
        return client.consumerPolicies.Get(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.GetConsumerPolicyPermissionDeniedException)