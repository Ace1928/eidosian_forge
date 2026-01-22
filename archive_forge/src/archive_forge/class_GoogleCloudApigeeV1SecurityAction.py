from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAction(_messages.Message):
    """A SecurityAction is rule that can be enforced at an environment level.
  The result is one of: - A denied API call - An explicitly allowed API call -
  A flagged API call (HTTP headers added before the target receives it) At
  least one condition is required to create a SecurityAction.

  Enums:
    StateValueValuesEnum: Required. Only an ENABLED SecurityAction is
      enforced. An ENABLED SecurityAction past its expiration time will not be
      enforced.

  Fields:
    allow: Allow a request through if it matches this SecurityAction.
    conditionConfig: Required. A valid SecurityAction must contain at least
      one condition.
    createTime: Output only. The create time for this SecurityAction.
    deny: Deny a request through if it matches this SecurityAction.
    description: Optional. An optional user provided description of the
      SecurityAction.
    expireTime: The expiration for this SecurityAction.
    flag: Flag a request through if it matches this SecurityAction.
    name: Immutable. This field is ignored during creation as per AIP-133.
      Please set the `security_action_id` field in the
      CreateSecurityActionRequest when creating a new SecurityAction. Format:
      organizations/{org}/environments/{env}/securityActions/{security_action}
    state: Required. Only an ENABLED SecurityAction is enforced. An ENABLED
      SecurityAction past its expiration time will not be enforced.
    ttl: Input only. The TTL for this SecurityAction.
    updateTime: Output only. The update time for this SecurityAction. This
      reflects when this SecurityAction changed states.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. Only an ENABLED SecurityAction is enforced. An ENABLED
    SecurityAction past its expiration time will not be enforced.

    Values:
      STATE_UNSPECIFIED: The default value. This only exists for forward
        compatibility. A create request with this value will be rejected.
      ENABLED: An ENABLED SecurityAction is actively enforced if the
        `expiration_time` is in the future.
      DISABLED: A disabled SecurityAction is never enforced.
    """
        STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
    allow = _messages.MessageField('GoogleCloudApigeeV1SecurityActionAllow', 1)
    conditionConfig = _messages.MessageField('GoogleCloudApigeeV1SecurityActionConditionConfig', 2)
    createTime = _messages.StringField(3)
    deny = _messages.MessageField('GoogleCloudApigeeV1SecurityActionDeny', 4)
    description = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    flag = _messages.MessageField('GoogleCloudApigeeV1SecurityActionFlag', 7)
    name = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    ttl = _messages.StringField(10)
    updateTime = _messages.StringField(11)