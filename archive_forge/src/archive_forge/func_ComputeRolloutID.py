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
def ComputeRolloutID(release_id, target_id, rollouts):
    """Generates a rollout ID.

  Args:
    release_id: str, release ID.
    target_id: str, target ID.
    rollouts: [apitools.base.protorpclite.messages.Message], list of rollout
      messages.

  Returns:
    rollout ID.

  Raises:
    googlecloudsdk.command_lib.deploy.exceptions.RolloutIdExhaustedError: if
    there are more than 1000 rollouts with auto-generated ID.
  """
    rollout_ids = {RolloutId(r.name) for r in rollouts}
    for i in range(1, 1001):
        rollout_id = ROLLOUT_ID_TEMPLATE.format(release_id, target_id, i)
        if rollout_id not in rollout_ids:
            return rollout_id
    raise cd_exceptions.RolloutIDExhaustedError(release_id)