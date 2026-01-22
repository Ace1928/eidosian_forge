from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DesiredDatapathProviderValueValuesEnum(_messages.Enum):
    """The desired datapath provider for the cluster.

    Values:
      DATAPATH_PROVIDER_UNSPECIFIED: Default value.
      LEGACY_DATAPATH: Use the IPTables implementation based on kube-proxy.
      ADVANCED_DATAPATH: Use the eBPF based GKE Dataplane V2 with additional
        features. See the [GKE Dataplane V2
        documentation](https://cloud.google.com/kubernetes-engine/docs/how-
        to/dataplane-v2) for more.
      MIGRATE_TO_ADVANCED_DATAPATH: Cluster has some existing nodes but new
        nodes should use ADVANCED_DATAPATH.
      MIGRATE_TO_LEGACY_DATAPATH: Cluster has some existing nodes but new
        nodes should use LEGACY_DATAPATH.
    """
    DATAPATH_PROVIDER_UNSPECIFIED = 0
    LEGACY_DATAPATH = 1
    ADVANCED_DATAPATH = 2
    MIGRATE_TO_ADVANCED_DATAPATH = 3
    MIGRATE_TO_LEGACY_DATAPATH = 4