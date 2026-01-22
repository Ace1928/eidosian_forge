from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterTypeValueValuesEnum(_messages.Enum):
    """Immutable. The on prem cluster's type.

    Values:
      CLUSTERTYPE_UNSPECIFIED: The ClusterType is not set.
      BOOTSTRAP: The ClusterType is bootstrap cluster.
      HYBRID: The ClusterType is baremetal hybrid cluster.
      STANDALONE: The ClusterType is baremetal standalone cluster.
      USER: The ClusterType is user cluster.
    """
    CLUSTERTYPE_UNSPECIFIED = 0
    BOOTSTRAP = 1
    HYBRID = 2
    STANDALONE = 3
    USER = 4