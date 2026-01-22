from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def ConstructAutoscalingSettingsMessage(settings_message_class, policy_message_class, thresholds_message_class, autoscaling_settings):
    """Constructs autoscaling settings API message.

  Args:
    settings_message_class: Top-level autoscaling settings message class.
    policy_message_class: Autoscaling policy message class.
    thresholds_message_class: Autoscaling policy thresholds message class.
    autoscaling_settings: Desired autoscaling settings.

  Returns:
    The constructed message.
  """
    if not autoscaling_settings:
        return None
    settings_message = settings_message_class()
    settings_message.minClusterNodeCount = autoscaling_settings.min_cluster_node_count
    settings_message.maxClusterNodeCount = autoscaling_settings.max_cluster_node_count
    settings_message.coolDownPeriod = autoscaling_settings.cool_down_period
    policy_messages = {}
    for name, policy in autoscaling_settings.autoscaling_policies.items():
        policy_message = policy_message_class()
        policy_message.nodeTypeId = policy.node_type_id
        policy_message.scaleOutSize = policy.scale_out_size
        policy_message.minNodeCount = policy.min_node_count
        policy_message.maxNodeCount = policy.max_node_count
        policy_message.cpuThresholds = _ConstructThresholdsMessage(policy.cpu_thresholds, thresholds_message_class)
        policy_message.grantedMemoryThresholds = _ConstructThresholdsMessage(policy.granted_memory_thresholds, thresholds_message_class)
        policy_message.consumedMemoryThresholds = _ConstructThresholdsMessage(policy.consumed_memory_thresholds, thresholds_message_class)
        policy_message.storageThresholds = _ConstructThresholdsMessage(policy.storage_thresholds, thresholds_message_class)
        policy_messages[name] = policy_message
    settings_message.autoscalingPolicies = settings_message_class.AutoscalingPoliciesValue(additionalProperties=[settings_message_class.AutoscalingPoliciesValue.AdditionalProperty(key=name, value=policy_message) for name, policy_message in policy_messages.items()])
    return settings_message