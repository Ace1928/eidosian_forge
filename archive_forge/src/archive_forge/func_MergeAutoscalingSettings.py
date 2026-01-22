from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def MergeAutoscalingSettings(left: AutoscalingSettings, right: AutoscalingSettings) -> AutoscalingSettings:
    """Merges two AutoscalingSettings objects, favoring right one.

  Args:
    left: First AutoscalingSettings object.
    right: Second AutoscalingSettings object.

  Returns:
    Merged AutoscalingSettings.
  """
    if left is None:
        return right
    if right is None:
        return left
    policies = {}
    for policy_name, policy in left.autoscaling_policies.items():
        if policy_name in right.autoscaling_policies:
            policies[policy_name] = _MergeAutoscalingPolicies(policy, right.autoscaling_policies[policy_name])
        else:
            policies[policy_name] = policy
    for policy_name, policy in right.autoscaling_policies.items():
        if policy_name not in left.autoscaling_policies:
            policies[policy_name] = policy
    return AutoscalingSettings(min_cluster_node_count=_MergeFields(left.min_cluster_node_count, right.min_cluster_node_count), max_cluster_node_count=_MergeFields(left.max_cluster_node_count, right.max_cluster_node_count), cool_down_period=_MergeFields(left.cool_down_period, right.cool_down_period), autoscaling_policies=policies)