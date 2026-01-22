from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelDescriptorsListRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelDescriptorsListRequest object.

  Fields:
    name: Required. The REST resource name of the parent from which to
      retrieve the notification channel descriptors. The expected syntax is:
      projects/[PROJECT_ID_OR_NUMBER] Note that this names
      (https://cloud.google.com/monitoring/api/v3#project_name) the parent
      container in which to look for the descriptors; to retrieve a single
      descriptor by name, use the GetNotificationChannelDescriptor operation,
      instead.
    pageSize: The maximum number of results to return in a single response. If
      not set to a positive number, a reasonable value will be chosen by the
      service.
    pageToken: If non-empty, page_token must contain a value returned as the
      next_page_token in a previous response to request the next set of
      results.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)