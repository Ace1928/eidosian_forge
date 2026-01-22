from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TagsValue(_messages.Message):
    """Optional. A set of tags to apply to all underlying Azure resources for
    this node pool. This currently only includes Virtual Machine Scale Sets.
    Specify at most 50 pairs containing alphanumerics, spaces, and symbols
    (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can be up to
    255 Unicode characters.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)