from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsSearchRequest
  object.

  Enums:
    ApplicationStatusValueValuesEnum: Optional. Search only applications in
      the chosen state.

  Fields:
    applicationStatus: Optional. Search only applications in the chosen state.
    maxEndTime: Optional. Latest end timestamp to list.
    maxTime: Optional. Latest start timestamp to list.
    minEndTime: Optional. Earliest end timestamp to list.
    minTime: Optional. Earliest start timestamp to list.
    pageSize: Optional. Maximum number of applications to return in each
      response. The service may return fewer than this. The default page size
      is 10; the maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplications call. Provide this token to retrieve the
      subsequent page.
    parent: Required. The fully qualified name of the session to retrieve in
      the format
      "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_ID"
  """

    class ApplicationStatusValueValuesEnum(_messages.Enum):
        """Optional. Search only applications in the chosen state.

    Values:
      APPLICATION_STATUS_UNSPECIFIED: <no description>
      APPLICATION_STATUS_RUNNING: <no description>
      APPLICATION_STATUS_COMPLETED: <no description>
    """
        APPLICATION_STATUS_UNSPECIFIED = 0
        APPLICATION_STATUS_RUNNING = 1
        APPLICATION_STATUS_COMPLETED = 2
    applicationStatus = _messages.EnumField('ApplicationStatusValueValuesEnum', 1)
    maxEndTime = _messages.StringField(2)
    maxTime = _messages.StringField(3)
    minEndTime = _messages.StringField(4)
    minTime = _messages.StringField(5)
    pageSize = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(7)
    parent = _messages.StringField(8, required=True)