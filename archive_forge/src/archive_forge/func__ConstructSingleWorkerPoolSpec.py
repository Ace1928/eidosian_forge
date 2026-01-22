from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _ConstructSingleWorkerPoolSpec(aiplatform_client, spec, python_package_uri=None):
    """Constructs the specification of a single worker pool.

  Args:
    aiplatform_client: The AI Platform API client used.
    spec: A dict whose fields represent a worker pool config.
    python_package_uri: str, The common python package uris that will be used by
      executor image, supposedly derived from the gcloud command flags.

  Returns:
    A WorkerPoolSpec message instance for setting a worker pool in a custom job.
  """
    worker_pool_spec = aiplatform_client.GetMessage('WorkerPoolSpec')()
    machine_spec_msg = aiplatform_client.GetMessage('MachineSpec')
    machine_spec = machine_spec_msg(machineType=spec.get('machine-type'))
    accelerator_type = spec.get('accelerator-type')
    if accelerator_type:
        machine_spec.acceleratorType = arg_utils.ChoiceToEnum(accelerator_type, machine_spec_msg.AcceleratorTypeValueValuesEnum)
        machine_spec.acceleratorCount = int(spec.get('accelerator-count', 1))
    worker_pool_spec.machineSpec = machine_spec
    worker_pool_spec.replicaCount = int(spec.get('replica-count', 1))
    container_image_uri = spec.get('container-image-uri')
    executor_image_uri = spec.get('executor-image-uri')
    python_module = spec.get('python-module')
    if container_image_uri:
        container_spec_msg = aiplatform_client.GetMessage('ContainerSpec')
        worker_pool_spec.containerSpec = container_spec_msg(imageUri=container_image_uri)
    elif python_package_uri or executor_image_uri or python_module:
        python_package_spec_msg = aiplatform_client.GetMessage('PythonPackageSpec')
        worker_pool_spec.pythonPackageSpec = python_package_spec_msg(executorImageUri=executor_image_uri, packageUris=python_package_uri or [], pythonModule=python_module)
    return worker_pool_spec