from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceCondition(_messages.Message):
    """ResourceCondition provides a standard mechanism for higher-level status
  reporting from controller.

  Enums:
    StateValueValuesEnum: state of the condition.

  Fields:
    lastTransitionTime: Last time the condition transit from one status to
      another.
    message: Human-readable message indicating details about last transition.
    reason: Machine-readable message indicating details about last transition.
    state: state of the condition.
    type: Type of the condition. (e.g., ClusterRunning, NodePoolRunning or
      ServerSidePreflightReady)
  """

    class StateValueValuesEnum(_messages.Enum):
        """state of the condition.

    Values:
      STATE_UNSPECIFIED: Not set.
      STATE_TRUE: Resource is in the condition.
      STATE_FALSE: Resource is not in the condition.
      STATE_UNKNOWN: Kubernetes controller can't decide if the resource is in
        the condition or not.
    """
        STATE_UNSPECIFIED = 0
        STATE_TRUE = 1
        STATE_FALSE = 2
        STATE_UNKNOWN = 3
    lastTransitionTime = _messages.StringField(1)
    message = _messages.StringField(2)
    reason = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    type = _messages.StringField(5)