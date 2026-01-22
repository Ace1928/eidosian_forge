from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def ParseAutoscalingSettingsFromFileFormat(cluster: Dict[str, Any]) -> AutoscalingSettings:
    """Parses the autoscaling settings from the format returned by  the describe command.

  The resulting object can later be passed to
  googlecloudsdk.api_lib.vmware.util.ConstructAutoscalingSettingsMessage.

  Args:
    cluster: dictionary with the settings. Parsed from a file provided by user.

  Returns:
    Equivalent AutoscalingSettings instance.

  Raises:
    InvalidAutoscalingSettingsProvidedError: if the file format was wrong.
  """

    def _ParseThresholds(thresholds_dict):
        if thresholds_dict is None:
            return None
        _ValidateIfOnlySupportedKeysArePassed(thresholds_dict.keys(), ['scaleIn', 'scaleOut'])
        return ScalingThresholds(scale_in=thresholds_dict.get('scaleIn'), scale_out=thresholds_dict.get('scaleOut'))
    _ValidateIfOnlySupportedKeysArePassed(cluster.keys(), ['autoscalingSettings'])
    if 'autoscalingSettings' not in cluster:
        raise InvalidAutoscalingSettingsProvidedError('autoscalingSettings not provided in the file')
    autoscaling_settings = cluster['autoscalingSettings']
    _ValidateIfOnlySupportedKeysArePassed(autoscaling_settings.keys(), ['minClusterNodeCount', 'maxClusterNodeCount', 'coolDownPeriod', 'autoscalingPolicies'])
    parsed_settings = AutoscalingSettings(min_cluster_node_count=autoscaling_settings.get('minClusterNodeCount'), max_cluster_node_count=autoscaling_settings.get('maxClusterNodeCount'), cool_down_period=autoscaling_settings.get('coolDownPeriod'), autoscaling_policies={})
    if 'autoscalingPolicies' not in autoscaling_settings:
        return parsed_settings
    for policy_name, policy_settings in autoscaling_settings['autoscalingPolicies'].items():
        _ValidateIfOnlySupportedKeysArePassed(policy_settings.keys(), ['nodeTypeId', 'scaleOutSize', 'minNodeCount', 'maxNodeCount', 'cpuThresholds', 'grantedMemoryThresholds', 'consumedMemoryThresholds', 'storageThresholds'])
        parsed_policy = AutoscalingPolicy(node_type_id=policy_settings.get('nodeTypeId'), scale_out_size=policy_settings.get('scaleOutSize'), min_node_count=policy_settings.get('minNodeCount'), max_node_count=policy_settings.get('maxNodeCount'), cpu_thresholds=_ParseThresholds(policy_settings.get('cpuThresholds')), granted_memory_thresholds=_ParseThresholds(policy_settings.get('grantedMemoryThresholds')), consumed_memory_thresholds=_ParseThresholds(policy_settings.get('consumedMemoryThresholds')), storage_thresholds=_ParseThresholds(policy_settings.get('storageThresholds')))
        parsed_settings.autoscaling_policies[policy_name] = parsed_policy
    return parsed_settings