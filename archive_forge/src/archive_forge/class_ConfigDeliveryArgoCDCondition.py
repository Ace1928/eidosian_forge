from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigDeliveryArgoCDCondition(_messages.Message):
    """Condition contains details for one aspect of the current state of the
  reconciliation object.

  Enums:
    StatusValueValuesEnum: status of the condition, one of True, False,
      Unknown.
    TypeValueValuesEnum: type of condition in CamelCase.

  Fields:
    lastTransitionTime: lastTransitionTime is the last time the condition
      transitioned from one status to another
    message: message is a human readable message indicating details about the
      transition. This may be an empty string.
    reason: reason contains a programmatic identifier indicating the reason
      for the condition's last transition.
    status: status of the condition, one of True, False, Unknown.
    type: type of condition in CamelCase.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """status of the condition, one of True, False, Unknown.

    Values:
      CONDITION_STATUS_UNSPECIFIED: CONDITION_STATUS_UNSPECIFIED is the
        default unspecified conditionStatus.
      TRUE: TRUE means a resource is in the condition.
      FALSE: FALSE means a resource is not in the condition.
      UNKNOWN: UNKNOWN means kubernetes can't decide if a resource is in the
        condition or not.
    """
        CONDITION_STATUS_UNSPECIFIED = 0
        TRUE = 1
        FALSE = 2
        UNKNOWN = 3

    class TypeValueValuesEnum(_messages.Enum):
        """type of condition in CamelCase.

    Values:
      CONDITION_TYPE_UNSPECIFIED: CONDITION_TYPE_UNSPECIFIED is the default
        unspecified conditionType.
      READY: READY indicates the type of the configdeliveryargocd' status
        condtion is "READY". This is a normally used term in k8s which used as
        a specific "conditionType". The "conditionStatus" tells the value of
        "READY" (e.g. conditionStatus=true means not ready).
    """
        CONDITION_TYPE_UNSPECIFIED = 0
        READY = 1
    lastTransitionTime = _messages.StringField(1)
    message = _messages.StringField(2)
    reason = _messages.StringField(3)
    status = _messages.EnumField('StatusValueValuesEnum', 4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)