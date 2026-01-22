from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsWriteRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsWriteRequest object.

  Fields:
    name: Required. The fully qualified name of the spark application to write
      data about in the format "projects/PROJECT_ID/locations/DATAPROC_REGION/
      sessions/SESSION_ID/sparkApplications/APPLICATION_ID"
    writeSessionSparkApplicationContextRequest: A
      WriteSessionSparkApplicationContextRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    writeSessionSparkApplicationContextRequest = _messages.MessageField('WriteSessionSparkApplicationContextRequest', 2)