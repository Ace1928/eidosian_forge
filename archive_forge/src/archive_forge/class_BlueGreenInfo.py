from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlueGreenInfo(_messages.Message):
    """Information relevant to blue-green upgrade.

  Enums:
    PhaseValueValuesEnum: Current blue-green upgrade phase.

  Fields:
    blueInstanceGroupUrls: The resource URLs of the [managed instance groups]
      (/compute/docs/instance-groups/creating-groups-of-managed-instances)
      associated with blue pool.
    bluePoolDeletionStartTime: Time to start deleting blue pool to complete
      blue-green upgrade, in [RFC3339](https://www.ietf.org/rfc/rfc3339.txt)
      text format.
    greenInstanceGroupUrls: The resource URLs of the [managed instance groups]
      (/compute/docs/instance-groups/creating-groups-of-managed-instances)
      associated with green pool.
    greenPoolVersion: Version of green pool.
    phase: Current blue-green upgrade phase.
  """

    class PhaseValueValuesEnum(_messages.Enum):
        """Current blue-green upgrade phase.

    Values:
      PHASE_UNSPECIFIED: Unspecified phase.
      UPDATE_STARTED: blue-green upgrade has been initiated.
      CREATING_GREEN_POOL: Start creating green pool nodes.
      CORDONING_BLUE_POOL: Start cordoning blue pool nodes.
      DRAINING_BLUE_POOL: Start draining blue pool nodes.
      NODE_POOL_SOAKING: Start soaking time after draining entire blue pool.
      DELETING_BLUE_POOL: Start deleting blue nodes.
      ROLLBACK_STARTED: Rollback has been initiated.
    """
        PHASE_UNSPECIFIED = 0
        UPDATE_STARTED = 1
        CREATING_GREEN_POOL = 2
        CORDONING_BLUE_POOL = 3
        DRAINING_BLUE_POOL = 4
        NODE_POOL_SOAKING = 5
        DELETING_BLUE_POOL = 6
        ROLLBACK_STARTED = 7
    blueInstanceGroupUrls = _messages.StringField(1, repeated=True)
    bluePoolDeletionStartTime = _messages.StringField(2)
    greenInstanceGroupUrls = _messages.StringField(3, repeated=True)
    greenPoolVersion = _messages.StringField(4)
    phase = _messages.EnumField('PhaseValueValuesEnum', 5)