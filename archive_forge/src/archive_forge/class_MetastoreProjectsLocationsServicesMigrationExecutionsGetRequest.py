from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesMigrationExecutionsGetRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesMigrationExecutionsGetRequest
  object.

  Fields:
    name: Required. The relative resource name of the migration execution to
      retrieve, in the following form:projects/{project_number}/locations/{loc
      ation_id}/services/{service_id}/migrationExecutions/{migration_execution
      _id}.
  """
    name = _messages.StringField(1, required=True)