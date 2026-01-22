from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsDeleteRequest object.

  Fields:
    etag: The current etag of the connection. If an etag is provided and does
      not match the current etag of the connection, deletion will be blocked
      and an ABORTED error will be returned.
    name: Required. The name of the Connection to delete. Format:
      `projects/*/locations/*/connections/*`.
    validateOnly: If set, validate the request, but do not actually post it.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)