from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ServiceIntegrationSpecsValue(_messages.Message):
    """Mapping of user defined keys to ServiceIntegrationSpec.

    Messages:
      AdditionalProperty: An additional property for a
        ServiceIntegrationSpecsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ServiceIntegrationSpecsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ServiceIntegrationSpecsValue object.

      Fields:
        key: Name of the additional property.
        value: A ServiceIntegrationSpec attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ServiceIntegrationSpec', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)