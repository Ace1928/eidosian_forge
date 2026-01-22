from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersGetRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalClustersGetRequest object.

  Enums:
    ViewValueValuesEnum: View for bare metal user cluster. When `BASIC` is
      specified, only the cluster resource name and admin cluster membership
      are returned. The default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the
      same as `FULL', which returns the complete cluster configuration
      details.

  Fields:
    allowMissing: Optional. If true, return BareMetal Cluster including the
      one that only exists in RMS.
    name: Required. Name of the bare metal user cluster to get. Format: "proje
      cts/{project}/locations/{location}/bareMetalClusters/{bare_metal_cluster
      }"
    view: View for bare metal user cluster. When `BASIC` is specified, only
      the cluster resource name and admin cluster membership are returned. The
      default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the same as `FULL',
      which returns the complete cluster configuration details.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View for bare metal user cluster. When `BASIC` is specified, only the
    cluster resource name and admin cluster membership are returned. The
    default/unset value `CLUSTER_VIEW_UNSPECIFIED` is the same as `FULL',
    which returns the complete cluster configuration details.

    Values:
      CLUSTER_VIEW_UNSPECIFIED: If the value is not set, the default `FULL`
        view is used.
      BASIC: Includes basic information of a cluster resource including
        cluster resource name and admin cluster membership.
      FULL: Includes the complete configuration for bare metal cluster
        resource. This is the default value for GetBareMetalClusterRequest
        method.
    """
        CLUSTER_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 3)