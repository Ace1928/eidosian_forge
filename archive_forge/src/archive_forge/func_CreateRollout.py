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
def CreateRollout(release_ref, to_target, rollout_id=None, annotations=None, labels=None, description=None, starting_phase_id=None, override_deploy_policies=None):
    """Creates a rollout by calling the rollout create API and waits for the operation to finish.

  Args:
    release_ref: protorpc.messages.Message, release resource object.
    to_target: str, the target to create create the rollout in.
    rollout_id: str, rollout ID.
    annotations: dict[str,str], a dict of annotation (key,value) pairs that
      allow clients to store small amounts of arbitrary data in cloud deploy
      resources.
    labels: dict[str,str], a dict of label (key,value) pairs that can be used to
      select cloud deploy resources and to find collections of cloud deploy
      resources that satisfy certain conditions.
    description: str, rollout description.
    starting_phase_id: str, rollout starting phase.
    override_deploy_policies: List of Deploy Policies to override.

  Raises:
      ListRolloutsError: an error occurred calling rollout list API.

  Returns:
    The rollout resource created.
  """
    final_rollout_id = rollout_id
    if not final_rollout_id:
        final_rollout_id = GenerateRolloutId(to_target, release_ref)
    resource_dict = release_ref.AsDict()
    rollout_ref = resources.REGISTRY.Parse(final_rollout_id, collection=_ROLLOUT_COLLECTION, params={'projectsId': resource_dict.get('projectsId'), 'locationsId': resource_dict.get('locationsId'), 'deliveryPipelinesId': resource_dict.get('deliveryPipelinesId'), 'releasesId': release_ref.Name()})
    rollout_obj = client_util.GetMessagesModule().Rollout(name=rollout_ref.RelativeName(), targetId=to_target, description=description)
    log.status.Print('Creating rollout {} in target {}'.format(rollout_ref.RelativeName(), to_target))
    operation = rollout.RolloutClient().Create(rollout_ref, rollout_obj, annotations, labels, starting_phase_id, override_deploy_policies)
    operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, collection='clouddeploy.projects.locations.operations')
    client_util.OperationsClient().WaitForOperation(operation, operation_ref, 'Waiting for rollout creation operation to complete')
    return rollout.RolloutClient().Get(rollout_ref.RelativeName())