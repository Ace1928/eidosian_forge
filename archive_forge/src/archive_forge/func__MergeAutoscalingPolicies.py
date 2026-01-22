from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _MergeAutoscalingPolicies(left: AutoscalingPolicy, right: AutoscalingPolicy) -> AutoscalingPolicy:
    """Merges two AutoscalingPolicy objects, favoring right one.

  Args:
    left: First AutoscalingPolicy object.
    right: Second AutoscalingPolicy object.

  Returns:
    Merged AutoscalingPolicy.
  """
    if left is None:
        return right
    if right is None:
        return left
    return AutoscalingPolicy(node_type_id=_MergeFields(left.node_type_id, right.node_type_id), scale_out_size=_MergeFields(left.scale_out_size, right.scale_out_size), min_node_count=_MergeFields(left.min_node_count, right.min_node_count), max_node_count=_MergeFields(left.max_node_count, right.max_node_count), cpu_thresholds=_MergeScalingThresholds(left.cpu_thresholds, right.cpu_thresholds), granted_memory_thresholds=_MergeScalingThresholds(left.granted_memory_thresholds, right.granted_memory_thresholds), consumed_memory_thresholds=_MergeScalingThresholds(left.consumed_memory_thresholds, right.consumed_memory_thresholds), storage_thresholds=_MergeScalingThresholds(left.storage_thresholds, right.storage_thresholds))