from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateMap(_messages.Message):
    """Defines a collection of certificate configurations.

  Messages:
    LabelsValue: Set of labels associated with a Certificate Map.

  Fields:
    createTime: Output only. The creation timestamp of a Certificate Map.
    description: One or more paragraphs of text description of a certificate
      map.
    gclbTargets: Output only. A list of GCLB targets which use this
      Certificate Map.
    labels: Set of labels associated with a Certificate Map.
    name: A user-defined name of the Certificate Map. Certificate Map names
      must be unique globally and match pattern
      `projects/*/locations/*/certificateMaps/*`.
    updateTime: Output only. The update timestamp of a Certificate Map.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with a Certificate Map.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    gclbTargets = _messages.MessageField('GclbTarget', 3, repeated=True)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)