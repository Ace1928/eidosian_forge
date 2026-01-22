from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ConstructResourcePoolSpecs(aiplatform_client, specs, **kwargs):
    """Constructs the resource pool specs for a persistent resource.

  Args:
    aiplatform_client: The AI Platform API client used.
    specs: A list of dict of resource pool specs, supposedly derived from
      the gcloud command flags.
    **kwargs: The keyword args to pass down to construct each worker pool spec.

  Returns:
    A list of ResourcePool message instances for creating a Persistent Resource.
  """
    resource_pool_specs = []
    for spec in specs:
        if spec:
            resource_pool_specs.append(_ConstructSingleResourcePoolSpec(aiplatform_client, spec, **kwargs))
        else:
            resource_pool_specs.append(aiplatform_client.GetMessage('ResourcePoolSpec')())
    return resource_pool_specs