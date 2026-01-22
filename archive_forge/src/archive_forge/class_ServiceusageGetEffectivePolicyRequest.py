from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageGetEffectivePolicyRequest(_messages.Message):
    """A ServiceusageGetEffectivePolicyRequest object.

  Enums:
    ViewValueValuesEnum: The view of the effective policy to use.

  Fields:
    name: Required. The name of the effective policy to retrieve. Format:
      `projects/100/effectivePolicy`, `folders/101/effectivePolicy`,
      `organizations/102/effectivePolicy`.
    view: The view of the effective policy to use.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view of the effective policy to use.

    Values:
      EFFECTIVE_POLICY_VIEW_UNSPECIFIED: The default / unset value. The API
        will default to the BASIC view.
      EFFECTIVE_POLICY_VIEW_BASIC: Include basic metadata about the effective
        policy, but not the source of policy rules. This is the default value.
      EFFECTIVE_POLICY_VIEW_FULL: Include everything.
    """
        EFFECTIVE_POLICY_VIEW_UNSPECIFIED = 0
        EFFECTIVE_POLICY_VIEW_BASIC = 1
        EFFECTIVE_POLICY_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)