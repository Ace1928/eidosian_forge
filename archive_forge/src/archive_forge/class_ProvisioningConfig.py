from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisioningConfig(_messages.Message):
    """A provisioning configuration.

  Enums:
    StateValueValuesEnum: Output only. State of ProvisioningConfig.

  Fields:
    cloudConsoleUri: Output only. URI to Cloud Console UI view of this
      provisioning config.
    customId: Optional. The user-defined identifier of the provisioning
      config.
    email: Email provided to send a confirmation with provisioning config to.
      Deprecated in favour of email field in request messages.
    handoverServiceAccount: A service account to enable customers to access
      instance credentials upon handover.
    instances: Instances to be created.
    location: Optional. Location name of this ProvisioningConfig. It is
      optional only for Intake UI transition period.
    name: Output only. The system-generated name of the provisioning config.
      This follows the UUID format.
    networks: Networks to be created.
    pod: Optional. Pod name. Pod is an independent part of infrastructure.
      Instance can be connected to the assets (networks, volumes, nfsshares)
      allocated in the same pod only.
    state: Output only. State of ProvisioningConfig.
    statusMessage: Optional status messages associated with the FAILED state.
    ticketId: A generated ticket id to track provisioning request.
    updateTime: Output only. Last update timestamp.
    volumes: Volumes to be created.
    vpcScEnabled: If true, VPC SC is enabled for the cluster.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of ProvisioningConfig.

    Values:
      STATE_UNSPECIFIED: State wasn't specified.
      DRAFT: ProvisioningConfig is a draft and can be freely modified.
      SUBMITTED: ProvisioningConfig was already submitted and cannot be
        modified.
      PROVISIONING: ProvisioningConfig was in the provisioning state.
        Initially this state comes from the work order table in big query when
        SNOW is used. Later this field can be set by the work order API.
      PROVISIONED: ProvisioningConfig was provisioned, meaning the resources
        exist.
      VALIDATED: ProvisioningConfig was validated. A validation tool will be
        run to set this state.
      CANCELLED: ProvisioningConfig was canceled.
      FAILED: The request is submitted for provisioning, with error return.
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        SUBMITTED = 2
        PROVISIONING = 3
        PROVISIONED = 4
        VALIDATED = 5
        CANCELLED = 6
        FAILED = 7
    cloudConsoleUri = _messages.StringField(1)
    customId = _messages.StringField(2)
    email = _messages.StringField(3)
    handoverServiceAccount = _messages.StringField(4)
    instances = _messages.MessageField('InstanceConfig', 5, repeated=True)
    location = _messages.StringField(6)
    name = _messages.StringField(7)
    networks = _messages.MessageField('NetworkConfig', 8, repeated=True)
    pod = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    statusMessage = _messages.StringField(11)
    ticketId = _messages.StringField(12)
    updateTime = _messages.StringField(13)
    volumes = _messages.MessageField('VolumeConfig', 14, repeated=True)
    vpcScEnabled = _messages.BooleanField(15)