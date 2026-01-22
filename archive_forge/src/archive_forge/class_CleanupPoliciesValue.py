from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CleanupPoliciesValue(_messages.Message):
    """Optional. Cleanup policies for this repository. Cleanup policies
    indicate when certain package versions can be automatically deleted. Map
    keys are policy IDs supplied by users during policy creation. They must
    unique within a repository and be under 128 characters in length.

    Messages:
      AdditionalProperty: An additional property for a CleanupPoliciesValue
        object.

    Fields:
      additionalProperties: Additional properties of type CleanupPoliciesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CleanupPoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A CleanupPolicy attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('CleanupPolicy', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)