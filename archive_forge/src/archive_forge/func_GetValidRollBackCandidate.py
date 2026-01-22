from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetValidRollBackCandidate(target_ref, pipeline_ref):
    """Gets the currently deployed release and the next valid release that can be rolled back to.

  Args:
    target_ref: protorpc.messages.Message, target resource object.
    pipeline_ref: protorpc.messages.Message, pipeline resource object.

  Raises:
      HttpException: an error occurred fetching a resource.

  Returns:
    An list containg the currently deployed release and the next valid
    deployable release.
  """
    iterable = GetFilteredRollouts(target_ref=target_ref, pipeline_ref=pipeline_ref, filter_str=DEPLOYED_ROLLOUT_FILTER_TEMPLATE, order_by=SUCCEED_ROLLOUT_ORDERBY, limit=None, page_size=10)
    rollouts = []
    for rollout_obj in iterable:
        if not rollouts:
            rollouts.append(rollout_obj)
        elif not _RolloutIsFromAbandonedRelease(rollout_obj):
            rollouts.append(rollout_obj)
        if len(rollouts) >= 2:
            break
    return rollouts