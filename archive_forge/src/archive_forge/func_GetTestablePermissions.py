from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def GetTestablePermissions(iam_client, messages, resource):
    """Returns the testable permissions for a resource.

  Args:
    iam_client: The iam client.
    messages: The iam messages.
    resource: Resource reference.

  Returns:
    List of permissions.
  """
    return list_pager.YieldFromList(iam_client.permissions, messages.QueryTestablePermissionsRequest(fullResourceName=iam_util.GetFullResourceName(resource), pageSize=1000), batch_size=1000, method='QueryTestablePermissions', field='permissions', batch_size_attribute='pageSize')