from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DerivedAssetConfig(_messages.Message):
    """A DerivedAssetConfig configures the associated AssetType to manage the
  derived assets for its Assets. The assets under the associated AssetType are
  owners of the derived assets. The derived assets are linked to the owners
  via the links in the owner assets.

  Messages:
    MetadataValue: Required. Key-value pairs for how to set the metadata in
      the derived assets. The key maps to the metadata in the derived assets.
      The value is interpreted as a literal or a path within the owner asset
      if it's prefixed by "$asset.", e.g. "$asset.file.url".

  Fields:
    metadata: Required. Key-value pairs for how to set the metadata in the
      derived assets. The key maps to the metadata in the derived assets. The
      value is interpreted as a literal or a path within the owner asset if
      it's prefixed by "$asset.", e.g. "$asset.file.url".
    owningLink: Required. The link in the owner asset.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Required. Key-value pairs for how to set the metadata in the derived
    assets. The key maps to the metadata in the derived assets. The value is
    interpreted as a literal or a path within the owner asset if it's prefixed
    by "$asset.", e.g. "$asset.file.url".

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    metadata = _messages.MessageField('MetadataValue', 1)
    owningLink = _messages.StringField(2)