from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ServiceLabelsValue(_messages.Message):
    """Labels that specify the resource that emits the monitoring data which
    is used for SLO reporting of this Service. Documentation and valid values
    for given service types here
    (https://cloud.google.com/stackdriver/docs/solutions/slo-
    monitoring/api/api-structures#basic-svc-w-basic-sli).

    Messages:
      AdditionalProperty: An additional property for a ServiceLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ServiceLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ServiceLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)