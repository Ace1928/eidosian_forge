from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterAutoscaling(_messages.Message):
    """ClusterAutoscaling contains global, per-cluster information required by
  Cluster Autoscaler to automatically adjust the size of the cluster and
  create/delete node pools based on the current needs.

  Enums:
    AutoscalingProfileValueValuesEnum: Defines autoscaling behaviour.

  Fields:
    autoprovisioningLocations: The list of Google Compute Engine
      [zones](https://cloud.google.com/compute/docs/zones#available) in which
      the NodePool's nodes can be created by NAP.
    autoprovisioningNodePoolDefaults: AutoprovisioningNodePoolDefaults
      contains defaults for a node pool created by NAP.
    autoscalingProfile: Defines autoscaling behaviour.
    enableNodeAutoprovisioning: Enables automatic node pool creation and
      deletion.
    resourceLimits: Contains global constraints regarding minimum and maximum
      amount of resources in the cluster.
  """

    class AutoscalingProfileValueValuesEnum(_messages.Enum):
        """Defines autoscaling behaviour.

    Values:
      PROFILE_UNSPECIFIED: No change to autoscaling configuration.
      OPTIMIZE_UTILIZATION: Prioritize optimizing utilization of resources.
      BALANCED: Use default (balanced) autoscaling configuration.
    """
        PROFILE_UNSPECIFIED = 0
        OPTIMIZE_UTILIZATION = 1
        BALANCED = 2
    autoprovisioningLocations = _messages.StringField(1, repeated=True)
    autoprovisioningNodePoolDefaults = _messages.MessageField('AutoprovisioningNodePoolDefaults', 2)
    autoscalingProfile = _messages.EnumField('AutoscalingProfileValueValuesEnum', 3)
    enableNodeAutoprovisioning = _messages.BooleanField(4)
    resourceLimits = _messages.MessageField('ResourceLimit', 5, repeated=True)