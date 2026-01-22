from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlPlanRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsAccessSqlPlanRequest
  object.

  Fields:
    executionId: Required. Execution ID
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    parent: Required. Parent (Session) resource reference.
  """
    executionId = _messages.IntegerField(1)
    name = _messages.StringField(2, required=True)
    parent = _messages.StringField(3)