from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvenanceBuilder(_messages.Message):
    """A ProvenanceBuilder object.

  Messages:
    VersionValue: A VersionValue object.

  Fields:
    builderDependencies: A ResourceDescriptor attribute.
    id: A string attribute.
    version: A VersionValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class VersionValue(_messages.Message):
        """A VersionValue object.

    Messages:
      AdditionalProperty: An additional property for a VersionValue object.

    Fields:
      additionalProperties: Additional properties of type VersionValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a VersionValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    builderDependencies = _messages.MessageField('ResourceDescriptor', 1, repeated=True)
    id = _messages.StringField(2)
    version = _messages.MessageField('VersionValue', 3)