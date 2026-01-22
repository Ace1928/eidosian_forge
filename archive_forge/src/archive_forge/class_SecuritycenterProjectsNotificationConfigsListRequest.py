from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsNotificationConfigsListRequest(_messages.Message):
    """A SecuritycenterProjectsNotificationConfigsListRequest object.

  Fields:
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last
      `ListNotificationConfigsResponse`; indicates that this is a continuation
      of a prior `ListNotificationConfigs` call, and that the system should
      return the next page of data.
    parent: Required. The name of the parent in which to list the notification
      configurations. Its format is "organizations/[organization_id]",
      "folders/[folder_id]", or "projects/[project_id]".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)