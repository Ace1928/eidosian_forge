from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Checks if there are any rollouts in progress for the given release.

  Args:
    release_ref: protorpc.messages.Message, release resource object.
    release_obj: apitools.base.protorpclite.messages.Message, release message
      object.
    to_target_id: string, target id (e.g. 'prod')

  Raises:
    googlecloudsdk.command_lib.deploy.exceptions.RolloutInProgressError
  