from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class IstioServiceIdentityAdmissionRulesValue(_messages.Message):
    """Optional. Per-istio-service-identity admission rules. Istio service
    identity spec format: `spiffe:///ns//sa/` or `/ns//sa/` e.g.
    `spiffe://example.com/ns/test-ns/sa/default`

    Messages:
      AdditionalProperty: An additional property for a
        IstioServiceIdentityAdmissionRulesValue object.

    Fields:
      additionalProperties: Additional properties of type
        IstioServiceIdentityAdmissionRulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a IstioServiceIdentityAdmissionRulesValue
      object.

      Fields:
        key: Name of the additional property.
        value: A AdmissionRule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AdmissionRule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)