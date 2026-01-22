from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import target
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetCurrentRollout(target_ref, pipeline_ref):
    """Gets the releases in the specified target and the last deployment associated with the target.

  Args:
    target_ref: protorpc.messages.Message, target resource object.
    pipeline_ref: protorpc.messages.Message, pipeline object.

  Returns:
    release messages associated with the target.
    last deployed rollout message.
  Raises:
   Exceptions raised by RolloutClient.GetCurrentRollout()
  """
    current_rollout = None
    try:
        rollouts = list(rollout_util.GetFilteredRollouts(target_ref, pipeline_ref, filter_str=rollout_util.DEPLOYED_ROLLOUT_FILTER_TEMPLATE, order_by=rollout_util.SUCCEED_ROLLOUT_ORDERBY, limit=1))
        if rollouts:
            current_rollout = rollouts[0]
    except apitools_exceptions.HttpError as error:
        log.debug('failed to get the current rollout of target {}: {}'.format(target_ref.RelativeName(), error.content))
    return current_rollout