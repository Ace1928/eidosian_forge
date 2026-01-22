from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ContactsValue(_messages.Message):
    """Output only. Map containing the points of contact for the given
    finding. The key represents the type of contact, while the value contains
    a list of all the contacts that pertain. Please refer to:
    https://cloud.google.com/resource-manager/docs/managing-notification-
    contacts#notification-categories { "security": { "contacts": [ { "email":
    "person1@company.com" }, { "email": "person2@company.com" } ] } }

    Messages:
      AdditionalProperty: An additional property for a ContactsValue object.

    Fields:
      additionalProperties: Additional properties of type ContactsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ContactsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudSecuritycenterV2ContactDetails attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudSecuritycenterV2ContactDetails', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)