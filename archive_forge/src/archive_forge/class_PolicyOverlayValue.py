from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PolicyOverlayValue(_messages.Message):
    """A mapping of the resources that you want to simulate policies for and
    the policies that you want to simulate. Keys are the full resource names
    for the resources. For example,
    `//cloudresourcemanager.googleapis.com/projects/my-project`. For examples
    of full resource names for Google Cloud services, see
    https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    Values are Policy objects representing the policies that you want to
    simulate. Replays automatically take into account any IAM policies
    inherited through the resource hierarchy, and any policies set on
    descendant resources. You do not need to include these policies in the
    policy overlay.

    Messages:
      AdditionalProperty: An additional property for a PolicyOverlayValue
        object.

    Fields:
      additionalProperties: Additional properties of type PolicyOverlayValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PolicyOverlayValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleIamV1Policy attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleIamV1Policy', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)