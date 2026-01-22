from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesMigrationExecutionsDeleteRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesMigrationExecutionsDeleteRequest
  object.

  Fields:
    name: Required. The relative resource name of the migrationExecution to
      delete, in the following form:projects/{project_number}/locations/{locat
      ion_id}/services/{service_id}/migrationExecutions/{migration_execution_i
      d}.
    requestId: Optional. A request ID. Specify a unique request ID to allow
      the server to ignore the request if it has completed. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 60 minutes after the first request.For example, if an initial
      request times out, followed by another request with the same request ID,
      the server ignores the second request to prevent the creation of
      duplicate commitments.The request ID must be a valid UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier#Format) A
      zero UUID (00000000-0000-0000-0000-000000000000) is not supported.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)