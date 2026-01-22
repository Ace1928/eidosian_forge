from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiDocDocumentation(_messages.Message):
    """The documentation for a catalog item.

  Fields:
    graphqlDocumentation: Optional. GraphQL documentation.
    oasDocumentation: Optional. OpenAPI Specification documentation.
  """
    graphqlDocumentation = _messages.MessageField('GoogleCloudApigeeV1GraphqlDocumentation', 1)
    oasDocumentation = _messages.MessageField('GoogleCloudApigeeV1OASDocumentation', 2)