from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexEntity(_messages.Message):
    """Index is not used as an independent entity, it is retrieved as part of a
  Table entity.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    customFeatures: Custom engine specific features.
    name: The name of the index.
    tableColumns: Table columns used as part of the Index, for example B-TREE
      index should list the columns which constitutes the index.
    type: Type of index, for example B-TREE.
    unique: Boolean value indicating whether the index is unique.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 1)
    name = _messages.StringField(2)
    tableColumns = _messages.StringField(3, repeated=True)
    type = _messages.StringField(4)
    unique = _messages.BooleanField(5)