from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Constructs the spec of a custom job to be used in job creation request.

  Args:
    aiplatform_client: The AI Platform API client used.
    base_config: A base CustomJobSpec message instance, e.g. imported from a
      YAML config file, as a template to be overridden.
    network: user network to which the job should be peered with (overrides yaml
      file)
    service_account: A service account (email address string) to use for the
      job.
    enable_web_access: Whether to enable the interactive shell for the job.
    enable_dashboard_access: Whether to enable the access to the dashboard built
      on the job.
    worker_pool_specs: A dict of worker pool specification, usually derived from
      the gcloud command argument values.
    args: A list of arguments to be passed to containers or python packge,
      supposedly derived from the gcloud command flags.
    command: A list of commands to be passed to containers, supposedly derived
      from the gcloud command flags.
    persistent_resource_id: The name of the persistent resource from the same
      project and region on which to run this custom job.
    **kwargs: The keyword args to pass to construct the worker pool specs.

  Returns:
    A CustomJobSpec message instance for creating a custom job.
  