from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesSchemasCreateRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesSchemasCreateRequest
  object.

  Fields:
    parent: Required. Value for parent.
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
    schema: A Schema resource to be passed as the request body.
    schemaId: Required. The ID to use for the schema, which will become the
      final component of the schema's resource name. Currently, only `main` is
      supported and any other schema ID will result in an error.
    validateOnly: Optional. If set, validate the request and preview the
      Schema, but do not actually update it.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    schema = _messages.MessageField('Schema', 3)
    schemaId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)