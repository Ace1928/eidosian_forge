from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _ConstructWorkerPoolSpecs(aiplatform_client, specs, **kwargs):
    """Constructs the specification of the worker pools in a CustomJobSpec instance.

  Args:
    aiplatform_client: The AI Platform API client used.
    specs: A list of dict of worker pool specifications, supposedly derived from
      the gcloud command flags.
    **kwargs: The keyword args to pass down to construct each worker pool spec.

  Returns:
    A list of WorkerPoolSpec message instances for creating a custom job.
  """
    worker_pool_specs = []
    for spec in specs:
        if spec:
            worker_pool_specs.append(_ConstructSingleWorkerPoolSpec(aiplatform_client, spec, **kwargs))
        else:
            worker_pool_specs.append(aiplatform_client.GetMessage('WorkerPoolSpec')())
    return worker_pool_specs