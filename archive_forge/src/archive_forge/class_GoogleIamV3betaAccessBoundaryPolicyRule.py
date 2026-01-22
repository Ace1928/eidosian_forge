from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaAccessBoundaryPolicyRule(_messages.Message):
    """Rule details inside a principal access boundary policy.

  Enums:
    ActionValueValuesEnum: Required. The action that all principals which are
      impacted by this policy can take on resources inside the boundary.

  Fields:
    action: Required. The action that all principals which are impacted by
      this policy can take on resources inside the boundary.
    description: Optional. A user-specified description of the rule. This
      value can be up to 256 characters.
    resources: Required. Cloud Resource Manager resource name. The resource
      and all the descendants are included. The list is limited to 10
      resources. This represents all the boundaries of the policy. The
      following resource names are supported: * Organization, such as
      "//cloudresourcemanager.googleapis.com/organizations/123". * Folder,
      such as "//cloudresourcemanager.googleapis.com/folders/123". * Project,
      such as "//cloudresourcemanager.googleapis.com/projects/123" or
      "//cloudresourcemanager.googleapis.com/projects/my-project-id".
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. The action that all principals which are impacted by this
    policy can take on resources inside the boundary.

    Values:
      ACTION_UNSPECIFIED: Action unspecified; not a valid state.
      ALLOW: Allows all principals access to the resources in this rule.
    """
        ACTION_UNSPECIFIED = 0
        ALLOW = 1
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    description = _messages.StringField(2)
    resources = _messages.StringField(3, repeated=True)