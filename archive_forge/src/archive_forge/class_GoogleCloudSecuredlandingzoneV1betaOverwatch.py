from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuredlandingzoneV1betaOverwatch(_messages.Message):
    """The Overwatch resource which holds all metadata related to an Overwatch
  instance.

  Enums:
    StateValueValuesEnum: Output only. The operation state of Overwatch
      resource that decides if response actions will be taken upon receiving
      drift or threat signals. This field is mutable by using the
      ActivateOverwatch or SuspendOverwatch actions.

  Fields:
    createTime: Output only. Creation time.
    name: Output only. The name of this Overwatch resource, in the format of
      organizations/{org_id}/locations/{location_id}/overwatches/{overwatch_id
      }.
    planData: Input only. The terraform plan file passed as bytes.
    remediationServiceAccount: Output only. This service account will be used
      by the Overwatch service for remediating drifts.
    state: Output only. The operation state of Overwatch resource that decides
      if response actions will be taken upon receiving drift or threat
      signals. This field is mutable by using the ActivateOverwatch or
      SuspendOverwatch actions.
    updateTime: Output only. Update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The operation state of Overwatch resource that decides if
    response actions will be taken upon receiving drift or threat signals.
    This field is mutable by using the ActivateOverwatch or SuspendOverwatch
    actions.

    Values:
      STATE_UNSPECIFIED: Unspecified operation state.
      SUSPENDED: The Overwatch resource is suspended and no response actions
        are taken.
      ACTIVE: The Overwatch resource is active, and response actions will be
        taken based on the policies, when signals are received. This is the
        normal operating state.
      CREATING: The Overwatch resource is being created and not yet active.
      DELETING: The Overwatch resource is in the process of being deleted.
      UPDATING: The Overwatch resource's blueprint state is being updated.
    """
        STATE_UNSPECIFIED = 0
        SUSPENDED = 1
        ACTIVE = 2
        CREATING = 3
        DELETING = 4
        UPDATING = 5
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    planData = _messages.BytesField(3)
    remediationServiceAccount = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    updateTime = _messages.StringField(6)