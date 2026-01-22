from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsConnectionsCreateRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsConnectionsCreateRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    connectionId: Optional. User-settable connection resource ID. * Must start
      with a letter. * Must contain between 4-63 characters from `/a-z-/`. *
      Must end with a number or a letter.
    parent: Required. The resource project name of the connection location
      using the form: `projects/{project_id}/locations/{location_id}`
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set, validates request by executing a dry-run
      which would not alter the resource in any way.
  """
    connection = _messages.MessageField('Connection', 1)
    connectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)