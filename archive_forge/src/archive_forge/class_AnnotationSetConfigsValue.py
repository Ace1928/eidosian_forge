from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AnnotationSetConfigsValue(_messages.Message):
    """Mapping of annotationSet ID to its configuration. The annotationSet ID
    will be used as the resource ID when GCMA creates the AnnotationSet
    internally. Detailed rules for a resource id are: 1. 1 character minimum,
    63 characters maximum 2. only contains letters, digits, underscore and
    hyphen 3. starts with a letter if length == 1, starts with a letter or
    underscore if length > 1

    Messages:
      AdditionalProperty: An additional property for a
        AnnotationSetConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AnnotationSetConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AnnotationSetConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A AnnotationSetConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AnnotationSetConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)