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
def GetFilteredRollouts(target_ref, pipeline_ref, filter_str, order_by, page_size=None, limit=None):
    """Gets successfully deployed rollouts for the releases associated with the specified target and index.

  Args:
    target_ref: protorpc.messages.Message, target object.
    pipeline_ref: protorpc.messages.Message, pipeline object.
    filter_str: Filter string to use when listing rollouts.
    order_by: order_by field to use when listing rollouts.
    page_size: int, the maximum number of objects to return per page.
    limit: int, the maximum number of `Rollout` objects to return.

  Returns:
    a rollout object or None if no rollouts in the target.
  """
    parent = WILDCARD_RELEASE_NAME_TEMPLATE.format(pipeline_ref.RelativeName())
    return rollout.RolloutClient().List(release_name=parent, filter_str=filter_str.format(target_ref.Name()), order_by=order_by, page_size=page_size, limit=limit)