from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshControlPlaneRevision(_messages.Message):
    """Status of a control plane revision that is intended to be available for
  use in the cluster.

  Enums:
    ChannelValueValuesEnum: Release Channel the managed control plane revision
      is subscribed to.
    StateValueValuesEnum: State of the control plane revision.
      LIFECYCLE_STATE_UNSPECIFIED, FAILED_PRECONDITION, PROVISIONING, ACTIVE,
      and STALLED are applicable here.
    TypeValueValuesEnum: Type of the control plane revision.

  Fields:
    channel: Release Channel the managed control plane revision is subscribed
      to.
    details: Explanation of the state.
    owner: Owner of the control plane revision.
    revision: Unique name of the control plane revision.
    state: State of the control plane revision. LIFECYCLE_STATE_UNSPECIFIED,
      FAILED_PRECONDITION, PROVISIONING, ACTIVE, and STALLED are applicable
      here.
    type: Type of the control plane revision.
    version: Static version of the control plane revision.
  """

    class ChannelValueValuesEnum(_messages.Enum):
        """Release Channel the managed control plane revision is subscribed to.

    Values:
      CHANNEL_UNSPECIFIED: Unspecified
      RAPID: RAPID channel is offered on an early access basis for customers
        who want to test new releases.
      REGULAR: REGULAR channel is intended for production users who want to
        take advantage of new features.
      STABLE: STABLE channel includes versions that are known to be stable and
        reliable in production.
    """
        CHANNEL_UNSPECIFIED = 0
        RAPID = 1
        REGULAR = 2
        STABLE = 3

    class StateValueValuesEnum(_messages.Enum):
        """State of the control plane revision. LIFECYCLE_STATE_UNSPECIFIED,
    FAILED_PRECONDITION, PROVISIONING, ACTIVE, and STALLED are applicable
    here.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: Unspecified
      DISABLED: DISABLED means that the component is not enabled.
      FAILED_PRECONDITION: FAILED_PRECONDITION means that provisioning cannot
        proceed because of some characteristic of the member cluster.
      PROVISIONING: PROVISIONING means that provisioning is in progress.
      ACTIVE: ACTIVE means that the component is ready for use.
      STALLED: STALLED means that provisioning could not be done.
      NEEDS_ATTENTION: NEEDS_ATTENTION means that the component is ready, but
        some user intervention is required. (For example that the user should
        migrate workloads to a new control plane revision.)
      DEGRADED: DEGRADED means that the component is ready, but operating in a
        degraded state.
    """
        LIFECYCLE_STATE_UNSPECIFIED = 0
        DISABLED = 1
        FAILED_PRECONDITION = 2
        PROVISIONING = 3
        ACTIVE = 4
        STALLED = 5
        NEEDS_ATTENTION = 6
        DEGRADED = 7

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the control plane revision.

    Values:
      CONTROL_PLANE_REVISION_TYPE_UNSPECIFIED: Unspecified.
      UNMANAGED: User-installed in-cluster control plane revision.
      MANAGED_SERVICE: Google-managed service running outside the cluster.
        Note: Google-managed control planes are independent per-cluster,
        regardless of whether the revision name is the same or not.
      MANAGED_LOCAL: Google-managed local control plane revision.
    """
        CONTROL_PLANE_REVISION_TYPE_UNSPECIFIED = 0
        UNMANAGED = 1
        MANAGED_SERVICE = 2
        MANAGED_LOCAL = 3
    channel = _messages.EnumField('ChannelValueValuesEnum', 1)
    details = _messages.MessageField('ServiceMeshStatusDetails', 2, repeated=True)
    owner = _messages.StringField(3)
    revision = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)
    version = _messages.StringField(7)