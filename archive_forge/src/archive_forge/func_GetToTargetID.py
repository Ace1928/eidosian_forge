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
def GetToTargetID(release_obj, is_create):
    """Get the to_target parameter for promote API.

  This checks the promotion sequence to get the next stage to promote the
  release to.

  Args:
    release_obj: apitools.base.protorpclite.messages.Message, release message.
    is_create: bool, if getting the target for release creation.

  Returns:
    the target ID.

  Raises:
    NoStagesError: if no pipeline stages exist in the release.
    ReleaseInactiveError: if this is not called during release creation and the
    specified release has no rollouts.
  """
    if not release_obj.deliveryPipelineSnapshot.serialPipeline.stages:
        raise exceptions.NoStagesError(release_obj.name)
    release_ref = resources.REGISTRY.ParseRelativeName(release_obj.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases')
    to_target = release_obj.deliveryPipelineSnapshot.serialPipeline.stages[0].targetId
    reversed_stages = list(reversed(release_obj.deliveryPipelineSnapshot.serialPipeline.stages))
    release_dict = release_ref.AsDict()
    for i, stage in enumerate(reversed_stages):
        target_ref = target_util.TargetReference(stage.targetId, release_dict['projectsId'], release_dict['locationsId'])
        current_rollout = target_util.GetCurrentRollout(target_ref, resources.REGISTRY.Parse(None, collection='clouddeploy.projects.locations.deliveryPipelines', params={'projectsId': release_dict['projectsId'], 'locationsId': release_dict['locationsId'], 'deliveryPipelinesId': release_dict['deliveryPipelinesId']}))
        if current_rollout:
            current_rollout_ref = resources.REGISTRY.Parse(current_rollout.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts')
            if current_rollout_ref.Parent().Name() == release_ref.Name():
                if i > 0:
                    to_target = reversed_stages[i - 1].targetId
                else:
                    log.status.Print(_LAST_TARGET_IN_SEQUENCE.format(release_ref.Name(), target_ref.Name(), release_ref.RelativeName(), target_ref.RelativeName()))
                    to_target = target_ref.RelativeName()
                break
    if to_target == release_obj.deliveryPipelineSnapshot.serialPipeline.stages[0].targetId and (not is_create):
        raise exceptions.ReleaseInactiveError()
    return target_util.TargetId(to_target)