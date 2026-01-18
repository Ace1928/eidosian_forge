from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
Terminates a job run.

    Args:
      name: Name of the JobRun. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}/rollouts/{rollout}/jobRuns/{jobRun}.
      override_deploy_policies: List of Deploy Policies to override.

    Returns:
      TerminateJobRunResponse message.
    