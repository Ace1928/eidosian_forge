from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeshConnectivity(_messages.Message):
    """Status of cross cluster load balancing between other clusters in the
  mesh.

  Enums:
    StateValueValuesEnum: LifecycleState of multicluster load balancing.

  Fields:
    details: Explanation of state.
    state: LifecycleState of multicluster load balancing.
  """

    class StateValueValuesEnum(_messages.Enum):
        """LifecycleState of multicluster load balancing.

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
    details = _messages.MessageField('StatusDetails', 1, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 2)