from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.core.util import files
def _ValidateHardwareInWorkerPoolSpecArgs(worker_pool_specs, api_version):
    """Validates the hardware related fields specified in `--worker-pool-spec` flags.

  Args:
    worker_pool_specs: List[dict], a list of worker pool specs specified in
      command line.
    api_version: str, the API version this command will interact with, either GA
      or BETA.
  """
    for spec in worker_pool_specs:
        if spec:
            if 'machine-type' not in spec:
                raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Key [machine-type] required in dict arg but not provided.')
            if 'accelerator-count' in spec and 'accelerator-type' not in spec:
                raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Key [accelerator-type] required as [accelerator-count] is specified.')
            accelerator_type = spec.get('accelerator-type', None)
            if accelerator_type:
                type_enum = api_util.GetMessage('MachineSpec', api_version).AcceleratorTypeValueValuesEnum
                valid_types = [type for type in type_enum.names() if type.startswith('NVIDIA') or type.startswith('TPU')]
                if accelerator_type not in valid_types:
                    raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Found invalid value of [accelerator-type]: {actual}. Available values are [{expected}].'.format(actual=accelerator_type, expected=', '.join((v for v in sorted(valid_types)))))