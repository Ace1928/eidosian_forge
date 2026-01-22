from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3SerializedCategory(_messages.Message):
    """Message representing one category when exported as a nested proto.

  Fields:
    childCategories: Children of the category if any.
    description: Description of the category. The length of the description is
      limited to 2000 bytes when encoded in UTF-8.
    displayName: Required. Display name of the category. Max 200 bytes when
      encoded in UTF-8.
  """
    childCategories = _messages.MessageField('GoogleCloudDatacatalogV1alpha3SerializedCategory', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)