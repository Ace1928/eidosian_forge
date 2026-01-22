from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlsaProvenanceZeroTwo(_messages.Message):
    """See full explanation of fields at slsa.dev/provenance/v0.2.

  Messages:
    BuildConfigValue: A BuildConfigValue object.

  Fields:
    buildConfig: A BuildConfigValue attribute.
    buildType: A string attribute.
    builder: A GrafeasV1SlsaProvenanceZeroTwoSlsaBuilder attribute.
    invocation: A GrafeasV1SlsaProvenanceZeroTwoSlsaInvocation attribute.
    materials: A GrafeasV1SlsaProvenanceZeroTwoSlsaMaterial attribute.
    metadata: A GrafeasV1SlsaProvenanceZeroTwoSlsaMetadata attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BuildConfigValue(_messages.Message):
        """A BuildConfigValue object.

    Messages:
      AdditionalProperty: An additional property for a BuildConfigValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BuildConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    buildConfig = _messages.MessageField('BuildConfigValue', 1)
    buildType = _messages.StringField(2)
    builder = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaBuilder', 3)
    invocation = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaInvocation', 4)
    materials = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaMaterial', 5, repeated=True)
    metadata = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaMetadata', 6)