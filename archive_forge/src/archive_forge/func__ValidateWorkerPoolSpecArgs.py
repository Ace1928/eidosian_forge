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
def _ValidateWorkerPoolSpecArgs(worker_pool_specs, version):
    """Validates the argument values specified via `--worker-pool-spec` flags.

  Args:
    worker_pool_specs: List[dict], a list of worker pool specs specified in
      command line.
    version: str, the API version this command will interact with, either GA or
      BETA.
  """
    if not worker_pool_specs[0]:
        raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Empty value is not allowed for the first `--worker-pool-spec` flag.')
    _ValidateHardwareInWorkerPoolSpecArgs(worker_pool_specs, version)
    _ValidateSoftwareInWorkerPoolSpecArgs(worker_pool_specs)