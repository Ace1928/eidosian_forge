from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesBackupsCreateRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesBackupsCreateRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Required. The ID of the backup, which is used as the final
      component of the backup's name.This value must be between 1 and 64
      characters long, begin with a letter, end with a letter or number, and
      consist of alpha-numeric ASCII characters or hyphens.
    parent: Required. The relative resource name of the service in which to
      create a backup of the following form:projects/{project_number}/location
      s/{location_id}/services/{service_id}.
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
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)