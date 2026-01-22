from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesSetLabelsRequest(_messages.Message):
    """A InstancesSetLabelsRequest object.

  Messages:
    LabelsValue: A LabelsValue object.

  Fields:
    labelFingerprint: Fingerprint of the previous set of labels for this
      resource, used to prevent conflicts. Provide the latest fingerprint
      value when making a request to add or change labels.
    labels: A LabelsValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A LabelsValue object.

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
    labelFingerprint = _messages.BytesField(1)
    labels = _messages.MessageField('LabelsValue', 2)