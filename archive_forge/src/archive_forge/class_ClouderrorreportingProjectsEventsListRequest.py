from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouderrorreportingProjectsEventsListRequest(_messages.Message):
    """A ClouderrorreportingProjectsEventsListRequest object.

  Enums:
    TimeRangePeriodValueValuesEnum: Restricts the query to the specified time
      range.

  Fields:
    groupId: Required. The group for which events shall be returned. The
      `group_id` is a unique identifier for a particular error group. The
      identifier is derived from key parts of the error-log content and is
      treated as Service Data. For information about how Service Data is
      handled, see [Google Cloud Privacy
      Notice](https://cloud.google.com/terms/cloud-privacy-notice).
    pageSize: Optional. The maximum number of results to return per response.
    pageToken: Optional. A `next_page_token` provided by a previous response.
    projectName: Required. The resource name of the Google Cloud Platform
      project. Written as `projects/{projectID}`, where `{projectID}` is the
      [Google Cloud Platform project
      ID](https://support.google.com/cloud/answer/6158840). Example:
      `projects/my-project-123`.
    serviceFilter_resourceType: Optional. The exact value to match against
      [`ServiceContext.resource_type`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.resource_type).
    serviceFilter_service: Optional. The exact value to match against
      [`ServiceContext.service`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.service).
    serviceFilter_version: Optional. The exact value to match against
      [`ServiceContext.version`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.version).
    timeRange_period: Restricts the query to the specified time range.
  """

    class TimeRangePeriodValueValuesEnum(_messages.Enum):
        """Restricts the query to the specified time range.

    Values:
      PERIOD_UNSPECIFIED: Do not use.
      PERIOD_1_HOUR: Retrieve data for the last hour. Recommended minimum
        timed count duration: 1 min.
      PERIOD_6_HOURS: Retrieve data for the last 6 hours. Recommended minimum
        timed count duration: 10 min.
      PERIOD_1_DAY: Retrieve data for the last day. Recommended minimum timed
        count duration: 1 hour.
      PERIOD_1_WEEK: Retrieve data for the last week. Recommended minimum
        timed count duration: 6 hours.
      PERIOD_30_DAYS: Retrieve data for the last 30 days. Recommended minimum
        timed count duration: 1 day.
    """
        PERIOD_UNSPECIFIED = 0
        PERIOD_1_HOUR = 1
        PERIOD_6_HOURS = 2
        PERIOD_1_DAY = 3
        PERIOD_1_WEEK = 4
        PERIOD_30_DAYS = 5
    groupId = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectName = _messages.StringField(4, required=True)
    serviceFilter_resourceType = _messages.StringField(5)
    serviceFilter_service = _messages.StringField(6)
    serviceFilter_version = _messages.StringField(7)
    timeRange_period = _messages.EnumField('TimeRangePeriodValueValuesEnum', 8)