from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetPipelinesAndRollouts(target_ref, pipeline_refs):
    """Retrieves the latest rollout for each delivery pipeline.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_refs: protorpc.messages.Message, pipeline object.

  Returns:
    A list with element [pipeline_ref, rollout] where the rollout is the latest
    successful rollout of the pipeline resource.

  """
    result = []
    for pipeline_ref in pipeline_refs:
        rollout = target_util.GetCurrentRollout(target_ref, pipeline_ref)
        if rollout is not None:
            result.append([pipeline_ref, rollout])
    return result