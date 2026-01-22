from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationSetConfig(_messages.Message):
    """Configuration of an annotationSet.

  Messages:
    IndexedFieldConfigsValue: List of indexed fields (e.g. "data.start") to
      make available in searches with their corresponding properties.

  Fields:
    complexType: Required. Reference to the complex type name, in the
      following form:
      `projects/{project}/locations/{location}/complexTypes/{name}`. Complex
      type of the annotation set config has the following requirements: 1.
      Must have two required fields named start and end. 2. Allowed types for
      start and end: Video asset type: timecode. 3. Start and end should have
      the same type.
    indexedFieldConfigs: List of indexed fields (e.g. "data.start") to make
      available in searches with their corresponding properties.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class IndexedFieldConfigsValue(_messages.Message):
        """List of indexed fields (e.g. "data.start") to make available in
    searches with their corresponding properties.

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
        value: A IndexedFieldConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IndexedFieldConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    complexType = _messages.StringField(1)
    indexedFieldConfigs = _messages.MessageField('IndexedFieldConfigsValue', 2)