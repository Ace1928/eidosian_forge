from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FreeInstanceMetadata(_messages.Message):
    """Free instance specific metadata that is kept even after an instance has
  been upgraded for tracking purposes.

  Enums:
    ExpireBehaviorValueValuesEnum: Specifies the expiration behavior of a free
      instance. The default of ExpireBehavior is `REMOVE_AFTER_GRACE_PERIOD`.
      This can be modified during or after creation, and before expiration.

  Fields:
    expireBehavior: Specifies the expiration behavior of a free instance. The
      default of ExpireBehavior is `REMOVE_AFTER_GRACE_PERIOD`. This can be
      modified during or after creation, and before expiration.
    expireTime: Output only. Timestamp after which the instance will either be
      upgraded or scheduled for deletion after a grace period. ExpireBehavior
      is used to choose between upgrading or scheduling the free instance for
      deletion. This timestamp is set during the creation of a free instance.
    upgradeTime: Output only. If present, the timestamp at which the free
      instance was upgraded to a provisioned instance.
  """

    class ExpireBehaviorValueValuesEnum(_messages.Enum):
        """Specifies the expiration behavior of a free instance. The default of
    ExpireBehavior is `REMOVE_AFTER_GRACE_PERIOD`. This can be modified during
    or after creation, and before expiration.

    Values:
      EXPIRE_BEHAVIOR_UNSPECIFIED: Not specified.
      FREE_TO_PROVISIONED: When the free instance expires, upgrade the
        instance to a provisioned instance.
      REMOVE_AFTER_GRACE_PERIOD: When the free instance expires, disable the
        instance, and delete it after the grace period passes if it has not
        been upgraded.
    """
        EXPIRE_BEHAVIOR_UNSPECIFIED = 0
        FREE_TO_PROVISIONED = 1
        REMOVE_AFTER_GRACE_PERIOD = 2
    expireBehavior = _messages.EnumField('ExpireBehaviorValueValuesEnum', 1)
    expireTime = _messages.StringField(2)
    upgradeTime = _messages.StringField(3)