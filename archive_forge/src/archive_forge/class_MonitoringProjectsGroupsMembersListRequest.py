from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsMembersListRequest(_messages.Message):
    """A MonitoringProjectsGroupsMembersListRequest object.

  Fields:
    filter: An optional list filter
      (https://cloud.google.com/monitoring/api/learn_more#filtering)
      describing the members to be returned. The filter may reference the
      type, labels, and metadata of monitored resources that comprise the
      group. For example, to return only resources representing Compute Engine
      VM instances, use this filter: `resource.type = "gce_instance"`
    interval_endTime: Required. The end of the time interval.
    interval_startTime: Optional. The beginning of the time interval. The
      default value for the start time is the end time. The start time must
      not be later than the end time.
    name: Required. The group whose members are listed. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
    pageSize: A positive number that is the maximum number of results to
      return.
    pageToken: If this field is not empty then it must contain the
      next_page_token value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
  """
    filter = _messages.StringField(1)
    interval_endTime = _messages.StringField(2)
    interval_startTime = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(6)