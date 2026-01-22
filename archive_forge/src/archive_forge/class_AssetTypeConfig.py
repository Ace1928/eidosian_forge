from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssetTypeConfig(_messages.Message):
    """Catalog search item that includes the asset type and it's configuration.

  Enums:
    StateValueValuesEnum: Output only. Current state of the asset type config.

  Messages:
    IndexedFieldConfigsValue: A map between user-defined key and
      IndexedFieldConfig, where the key is indexed with the value configured
      by IndexedFieldConfig and customers can use the key as search operator
      if the expression is nonempty. If the expression in
      CatalogIndexedFieldConfig is empty, the key must be "".

  Fields:
    assetType: The asset type name that this catalog item configured. Format:
      projects/{project}/locations/{location}/assetTypes/{assetType}
    indexedFieldConfigs: A map between user-defined key and
      IndexedFieldConfig, where the key is indexed with the value configured
      by IndexedFieldConfig and customers can use the key as search operator
      if the expression is nonempty. If the expression in
      CatalogIndexedFieldConfig is empty, the key must be "".
    state: Output only. Current state of the asset type config.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the asset type config.

    Values:
      STATE_UNSPECIFIED: Unknown state.
      CREATING: Asset type config is being created.
      ACTIVE: Asset type config is active.
      DELETING: Asset type config is being deleted and will be automatically
        removed from catalog when the background deletion is done.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class IndexedFieldConfigsValue(_messages.Message):
        """A map between user-defined key and IndexedFieldConfig, where the key
    is indexed with the value configured by IndexedFieldConfig and customers
    can use the key as search operator if the expression is nonempty. If the
    expression in CatalogIndexedFieldConfig is empty, the key must be "".

    Messages:
      AdditionalProperty: An additional property for a
        IndexedFieldConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        IndexedFieldConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a IndexedFieldConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A CatalogIndexedFieldConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('CatalogIndexedFieldConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    assetType = _messages.StringField(1)
    indexedFieldConfigs = _messages.MessageField('IndexedFieldConfigsValue', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)