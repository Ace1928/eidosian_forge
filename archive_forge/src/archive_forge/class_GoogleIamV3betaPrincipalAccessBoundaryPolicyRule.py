from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaPrincipalAccessBoundaryPolicyRule(_messages.Message):
    """Principal access boundary policy rule that defines the resource
  boundary.

  Enums:
    EffectValueValuesEnum: Required. The access relationship of principals to
      the resources in this rule.

  Fields:
    description: Optional. The description of the principal access boundary
      policy rule. Must be less than or equal to 256 characters.
    effect: Required. The access relationship of principals to the resources
      in this rule.
    resources: Required. A list of Cloud Resource Manager resources. The
      resource and all the descendants are included. The following resource
      names are supported: * Organization, such as
      "//cloudresourcemanager.googleapis.com/organizations/123". * Folder,
      such as "//cloudresourcemanager.googleapis.com/folders/123". * Project,
      such as "//cloudresourcemanager.googleapis.com/projects/123" or
      "//cloudresourcemanager.googleapis.com/projects/my-project-id".
  """

    class EffectValueValuesEnum(_messages.Enum):
        """Required. The access relationship of principals to the resources in
    this rule.

    Values:
      EFFECT_UNSPECIFIED: Effect unspecified.
      ALLOW: Allows access to the resources in this rule.
    """
        EFFECT_UNSPECIFIED = 0
        ALLOW = 1
    description = _messages.StringField(1)
    effect = _messages.EnumField('EffectValueValuesEnum', 2)
    resources = _messages.StringField(3, repeated=True)