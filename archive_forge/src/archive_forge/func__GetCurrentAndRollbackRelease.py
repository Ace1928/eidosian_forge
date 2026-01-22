from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import promote_util
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.command_lib.deploy import rollout_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _GetCurrentAndRollbackRelease(release_id, pipeline_ref, target_ref):
    """Gets the current deployed release and the release that will be used by promote API to create the rollback rollout."""
    if release_id:
        ref_dict = target_ref.AsDict()
        current_rollout = target_util.GetCurrentRollout(target_ref, pipeline_ref)
        current_release_ref = resources.REGISTRY.ParseRelativeName(resources.REGISTRY.Parse(current_rollout.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts').Parent().RelativeName(), collection='clouddeploy.projects.locations.deliveryPipelines.releases')
        rollback_release_ref = resources.REGISTRY.Parse(release_id, collection='clouddeploy.projects.locations.deliveryPipelines.releases', params={'projectsId': ref_dict['projectsId'], 'locationsId': ref_dict['locationsId'], 'deliveryPipelinesId': pipeline_ref.Name(), 'releasesId': release_id})
        return (current_release_ref, rollback_release_ref)
    else:
        prior_rollouts = rollout_util.GetValidRollBackCandidate(target_ref, pipeline_ref)
        if len(prior_rollouts) < 2:
            raise core_exceptions.Error('unable to rollback target {}. Target has less than 2 rollouts.'.format(target_ref.Name()))
        current_deployed_rollout, previous_deployed_rollout = prior_rollouts
        current_release_ref = resources.REGISTRY.ParseRelativeName(resources.REGISTRY.Parse(current_deployed_rollout.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts').Parent().RelativeName(), collection='clouddeploy.projects.locations.deliveryPipelines.releases')
        rollback_release_ref = resources.REGISTRY.ParseRelativeName(resources.REGISTRY.Parse(previous_deployed_rollout.name, collection='clouddeploy.projects.locations.deliveryPipelines.releases.rollouts').Parent().RelativeName(), collection='clouddeploy.projects.locations.deliveryPipelines.releases')
        return (current_release_ref, rollback_release_ref)