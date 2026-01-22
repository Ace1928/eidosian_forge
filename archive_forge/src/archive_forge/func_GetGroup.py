from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
def GetGroup(version, group):
    """Get a Cloud Identity Group.

  Args:
    version: Release track information.
    group: Name of group as returned by LookupGroupName()
      (i.e. 'groups/{group_id}').
  Returns:
    Group resource object.
  """
    client = GetClient(version)
    messages = GetMessages(version)
    return client.groups.Get(messages.CloudidentityGroupsGetRequest(name=group))