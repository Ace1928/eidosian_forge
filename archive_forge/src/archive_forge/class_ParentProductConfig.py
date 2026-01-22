from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParentProductConfig(_messages.Message):
    """ParentProductConfig is the configuration of the parent product of the
  cluster. This field is used by Google internal products that are built on
  top of a GKE cluster and take the ownership of the cluster.

  Messages:
    LabelsValue: Labels contain the configuration of the parent product.

  Fields:
    labels: Labels contain the configuration of the parent product.
    productName: Name of the parent product associated with the cluster.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels contain the configuration of the parent product.

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
    labels = _messages.MessageField('LabelsValue', 1)
    productName = _messages.StringField(2)