from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaInstance(_messages.Message):
    """Instance conceptually encapsulates all Remote Build Execution resources
  for remote builds. An instance consists of storage and compute resources
  (for example, `ContentAddressableStorage`, `ActionCache`, `WorkerPools`)
  used for running remote builds. All Remote Build Execution API calls are
  scoped to an instance.

  Enums:
    StateValueValuesEnum: Output only. State of the instance.

  Fields:
    featurePolicy: The policy to define whether or not RBE features can be
      used or how they can be used.
    location: The location is a GCP region. Currently only `us-central1` is
      supported.
    loggingEnabled: Output only. Whether stack driver logging is enabled for
      the instance.
    name: Output only. Instance resource name formatted as:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`. Name should not be
      populated when creating an instance since it is provided in the
      `instance_id` field.
    schedulerNotificationConfig: The instance's configuration for scheduler
      notifications. Absence implies that this feature is not enabled for this
      instance.
    state: Output only. State of the instance.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the instance.

    Values:
      STATE_UNSPECIFIED: Not a valid state, but the default value of the enum.
      CREATING: The instance is in state `CREATING` once `CreateInstance` is
        called and before the instance is ready for use.
      RUNNING: The instance is in state `RUNNING` when it is ready for use.
      INACTIVE: An `INACTIVE` instance indicates that there is a problem that
        needs to be fixed. Such instances cannot be used for execution and
        instances that remain in this state for a significant period of time
        will be removed permanently.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        RUNNING = 2
        INACTIVE = 3
    featurePolicy = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicy', 1)
    location = _messages.StringField(2)
    loggingEnabled = _messages.BooleanField(3)
    name = _messages.StringField(4)
    schedulerNotificationConfig = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaSchedulerNotificationConfig', 5)
    state = _messages.EnumField('StateValueValuesEnum', 6)