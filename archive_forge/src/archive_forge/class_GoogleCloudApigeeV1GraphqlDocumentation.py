from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GraphqlDocumentation(_messages.Message):
    """GraphQL documentation for a catalog item.

  Fields:
    endpointUri: Required. The GraphQL endpoint URI to be queried by API
      consumers. Max length is 2,083 characters.
    schema: Required. The documentation file contents for the GraphQL schema.
  """
    endpointUri = _messages.StringField(1)
    schema = _messages.MessageField('GoogleCloudApigeeV1DocumentationFile', 2)