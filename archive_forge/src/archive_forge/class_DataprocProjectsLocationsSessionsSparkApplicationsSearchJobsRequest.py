from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchJobsRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsSearchJobsRequest
  object.

  Enums:
    JobStatusValueValuesEnum: Optional. List only jobs in the specific state.

  Fields:
    jobStatus: Optional. List only jobs in the specific state.
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of jobs to return in each response. The
      service may return fewer than this. The default page size is 10; the
      maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplicationJobs call. Provide this token to retrieve
      the subsequent page.
    parent: Required. Parent (Session) resource reference.
  """

    class JobStatusValueValuesEnum(_messages.Enum):
        """Optional. List only jobs in the specific state.

    Values:
      JOB_EXECUTION_STATUS_UNSPECIFIED: <no description>
      JOB_EXECUTION_STATUS_RUNNING: <no description>
      JOB_EXECUTION_STATUS_SUCCEEDED: <no description>
      JOB_EXECUTION_STATUS_FAILED: <no description>
      JOB_EXECUTION_STATUS_UNKNOWN: <no description>
    """
        JOB_EXECUTION_STATUS_UNSPECIFIED = 0
        JOB_EXECUTION_STATUS_RUNNING = 1
        JOB_EXECUTION_STATUS_SUCCEEDED = 2
        JOB_EXECUTION_STATUS_FAILED = 3
        JOB_EXECUTION_STATUS_UNKNOWN = 4
    jobStatus = _messages.EnumField('JobStatusValueValuesEnum', 1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5)