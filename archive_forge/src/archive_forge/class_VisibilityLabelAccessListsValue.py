from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class VisibilityLabelAccessListsValue(_messages.Message):
    """ACLs for access to restricted parts of the service.  The map key is
    the visibility label that is being controlled.  Note that access to any
    label also implies access to the unrestricted surface.

    Messages:
      AdditionalProperty: An additional property for a
        VisibilityLabelAccessListsValue object.

    Fields:
      additionalProperties: Additional properties of type
        VisibilityLabelAccessListsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a VisibilityLabelAccessListsValue object.

      Fields:
        key: Name of the additional property.
        value: A ServiceAccessList attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ServiceAccessList', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)