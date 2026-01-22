from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the connection is not found a new
      connection will be created. In this situation `update_mask` is ignored.
      The creation will succeed only if the input connection has all the
      necessary information (e.g a github_config with both user_oauth_token
      and installation_id properties).
    connection: A Connection resource to be passed as the request body.
    etag: The current etag of the connection. If an etag is provided and does
      not match the current etag of the connection, update will be blocked and
      an ABORTED error will be returned.
    name: Immutable. The resource name of the connection, in the format
      `projects/{project}/locations/{location}/connections/{connection_id}`.
    updateMask: The list of fields to be updated.
  """
    allowMissing = _messages.BooleanField(1)
    connection = _messages.MessageField('Connection', 2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    updateMask = _messages.StringField(5)