from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsableSubnetworkSecondaryRange(_messages.Message):
    """Secondary IP range of a usable subnetwork.

  Enums:
    StatusValueValuesEnum: This field is to determine the status of the
      secondary range programmably.

  Fields:
    ipCidrRange: The range of IP addresses belonging to this subnetwork
      secondary range.
    rangeName: The name associated with this subnetwork secondary range, used
      when adding an alias IP range to a VM instance.
    status: This field is to determine the status of the secondary range
      programmably.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """This field is to determine the status of the secondary range
    programmably.

    Values:
      UNKNOWN: UNKNOWN is the zero value of the Status enum. It's not a valid
        status.
      UNUSED: UNUSED denotes that this range is unclaimed by any cluster.
      IN_USE_SERVICE: IN_USE_SERVICE denotes that this range is claimed by
        cluster(s) for services. User-managed services range can be shared
        between clusters within the same subnetwork.
      IN_USE_SHAREABLE_POD: IN_USE_SHAREABLE_POD denotes this range was
        created by the network admin and is currently claimed by a cluster for
        pods. It can only be used by other clusters as a pod range.
      IN_USE_MANAGED_POD: IN_USE_MANAGED_POD denotes this range was created by
        GKE and is claimed for pods. It cannot be used for other clusters.
    """
        UNKNOWN = 0
        UNUSED = 1
        IN_USE_SERVICE = 2
        IN_USE_SHAREABLE_POD = 3
        IN_USE_MANAGED_POD = 4
    ipCidrRange = _messages.StringField(1)
    rangeName = _messages.StringField(2)
    status = _messages.EnumField('StatusValueValuesEnum', 3)