from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesPolicyTagsPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesPolicyTagsPatchRequest object.

  Fields:
    googleCloudDatacatalogV1PolicyTag: A GoogleCloudDatacatalogV1PolicyTag
      resource to be passed as the request body.
    name: Identifier. Resource name of this policy tag in the URL format. The
      policy tag manager generates unique taxonomy IDs and policy tag IDs.
    updateMask: Specifies the fields to update. You can update only display
      name, description, and parent policy tag. If not set, defaults to all
      updatable fields. For more information, see [FieldMask]
      (https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask).
  """
    googleCloudDatacatalogV1PolicyTag = _messages.MessageField('GoogleCloudDatacatalogV1PolicyTag', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)