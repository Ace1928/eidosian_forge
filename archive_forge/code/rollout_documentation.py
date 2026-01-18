from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.core import log
Calls the IgnoreJob API to ignore a job on a rollout within a specified phase.

    Args:
      name: Name of the Rollout. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}/rollouts/{rollout}.
      job: The job id on the rollout resource.
      phase: The phase id on the rollout resource.
      override_deploy_policies: List of Deploy Policies to override.

    Returns:
      IgnoreJobResponse message.
    