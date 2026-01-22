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
def _ValidateSoftwareInRestWorkerPoolSpecs(specs, is_local_package_specified=False):
    """Validates the argument values specified in all but the first `--worker-pool-spec` flags.

  Args:
    specs: List[dict], the list all but the first worker pool specs specified in
      command line.
    is_local_package_specified: bool, whether local package is specified
      in the first worker pool.
  """
    for spec in specs:
        if spec:
            if is_local_package_specified:
                software_fields = {'executor-image-uri', 'container-image-uri', 'python-module', 'script', 'requirements', 'extra-packages', 'extra-dirs'}
                _RaiseErrorIfUnexpectedKeys(unexpected_keys=software_fields.intersection(spec.keys()), reason='A local package has been specified in the first `--worker-pool-spec` flag and to be used for all workers, do not specify these keys elsewhere.')
            else:
                if 'local-package-path' in spec:
                    raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Key [local-package-path] is only allowed in the first `--worker-pool-spec` flag.')
                _ValidateWorkerPoolSoftwareWithoutLocalPackages(spec)