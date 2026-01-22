from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsCreateRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    connectionId: Required. The ID to use for the Connection, which will
      become the final component of the Connection's resource name. Names must
      be unique per-project per-location. Allows alphanumeric characters and
      any of -._~%!$&'()*+,;=@.
    parent: Required. Project and location where the connection will be
      created. Format: `projects/*/locations/*`.
  """
    connection = _messages.MessageField('Connection', 1)
    connectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)