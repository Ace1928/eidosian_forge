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
def _ValidateSoftwareInFirstWorkerPoolSpec(spec):
    """Validates the software related fields specified in the first `--worker-pool-spec` flags.

  Args:
    spec: dict, the specification of the first worker pool.

  Returns:
    A boolean value whether a local package will be used.
  """
    if 'local-package-path' in spec:
        _ValidateWorkerPoolSoftwareWithLocalPackage(spec)
        return True
    else:
        _ValidateWorkerPoolSoftwareWithoutLocalPackages(spec)
        return False