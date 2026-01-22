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
def GetMostRecentlyActivePipeline(target_ref, sorted_pipeline_refs):
    """Retrieves latest rollout and release information for a list of delivery pipelines.

  Args:
    target_ref: protorpc.messages.Message, target object.
    sorted_pipeline_refs: protorpc.messages.Message, a list of pipeline objects,
      sorted in descending order by create time.

  Returns:
    A tuple of the pipeline with the most recent deploy time with
     latest rollout.

  """
    pipeline_rollouts = GetPipelinesAndRollouts(target_ref, sorted_pipeline_refs)
    if not pipeline_rollouts:
        log.debug('Target: {} has no recently active pipelines.'.format(target_ref.RelativeName()))
        return (sorted_pipeline_refs[0], None)
    most_recent_pipeline_ref, most_recent_rollout = pipeline_rollouts[0]
    most_recent_rollout_deploy_time = datetime.datetime.strptime(most_recent_rollout.deployEndTime, '%Y-%m-%dT%H:%M:%S.%fZ')
    for pipeline_rollout_tuple in pipeline_rollouts[1:]:
        pipeline_ref, rollout = pipeline_rollout_tuple
        rollout_deploy_time = datetime.datetime.strptime(rollout.deployEndTime, '%Y-%m-%dT%H:%M:%S.%fZ')
        if rollout_deploy_time > most_recent_rollout_deploy_time:
            most_recent_pipeline_ref = pipeline_ref
            most_recent_rollout = rollout
            most_recent_rollout_deploy_time = rollout_deploy_time
    return (most_recent_pipeline_ref, most_recent_rollout)