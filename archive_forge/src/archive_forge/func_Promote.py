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
def Promote(release_ref, release_obj, to_target, is_create, rollout_id=None, annotations=None, labels=None, description=None, starting_phase_id=None, override_deploy_policies=None):
    """Creates a rollout for the given release in the destination target.

  If to_target is not specified and there is a rollout in progress, the promote
  will fail. Otherwise this computes the destination target based on the
  promotion sequence.

  Args:
    release_ref: protorpc.messages.Message, release resource object.
    release_obj: apitools.base.protorpclite.messages.Message, release message
      object.
    to_target: str, the target to promote the release to.
    is_create: bool, if creates a rollout during release creation.
    rollout_id: str, ID to assign to the generated rollout.
    annotations: dict[str,str], a dict of annotation (key,value) pairs that
      allow clients to store small amounts of arbitrary data in cloud deploy
      resources.
    labels: dict[str,str], a dict of label (key,value) pairs that can be used to
      select cloud deploy resources and to find collections of cloud deploy
      resources that satisfy certain conditions.
    description: str, rollout description.
    starting_phase_id: str, the starting phase for the rollout.
    override_deploy_policies: List of Deploy Policies to override.

  Raises:
    googlecloudsdk.command_lib.deploy.exceptions.RolloutIdExhausted
    googlecloudsdk.command_lib.deploy.exceptions.ReleaseInactiveError
  Returns:
    The rollout resource created.
  """
    dest_target = to_target
    if not dest_target:
        dest_target = GetToTargetID(release_obj, is_create)
    rollout_resource = rollout_util.CreateRollout(release_ref, dest_target, rollout_id, annotations, labels, description, starting_phase_id, override_deploy_policies)
    target_obj = release_util.GetSnappedTarget(release_obj, dest_target)
    if target_obj and target_obj.requireApproval:
        log.status.Print('The rollout is pending approval.')
    return rollout_resource