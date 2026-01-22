from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ExternalSystemsValue(_messages.Message):
    """Output only. Third party SIEM/SOAR fields within SCC, contains
    external system information and external system finding fields.

    Messages:
      AdditionalProperty: An additional property for a ExternalSystemsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExternalSystemsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ExternalSystemsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudSecuritycenterV2ExternalSystem attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudSecuritycenterV2ExternalSystem', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)