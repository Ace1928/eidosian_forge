from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.resource_manager.exceptions import ResourceManagerError
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.core import exceptions as core_exceptions
def _GetServiceAccountUniqueId(service_account_email):
    """Returns the unique id for the given service account email.

  Args:
    service_account_email: email of the service account.

  Returns:
    The unique id of the service account.
  """
    client, messages = iam_api.GetClientAndMessages()
    try:
        res = client.projects_serviceAccounts.Get(messages.IamProjectsServiceAccountsGetRequest(name=iam_util.EmailToAccountResourceName(service_account_email)))
        return str(res.uniqueId)
    except apitools_exceptions.HttpError as e:
        raise exceptions.HttpException(e)