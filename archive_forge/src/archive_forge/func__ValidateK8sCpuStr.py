import re
import types
from typing import FrozenSet, Optional, Tuple
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import types as api_types
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import labels_util
from googlecloudsdk.command_lib.functions import run_util
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
def _ValidateK8sCpuStr(cpu: str) -> Optional[str]:
    """Validates user provided cpu to kubernetes expected format.

  k8s format:
  https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/api/resource/generated.proto

  Args:
    cpu: input from `args.cpu`

  Returns:
    k8s_cpu: str|None, in kubernetes cpu format.

  Raises:
    InvalidArgumentException: User provided invalid input for flag.
  """
    if cpu is None:
        return None
    match = re.match(_CPU_VALUE_PATTERN, cpu, re.VERBOSE)
    if not match:
        raise exceptions.InvalidArgumentException('--cpu', 'Invalid cpu value for: {} specified.'.format(cpu))
    suffix = match.group('suffix') or ''
    amount = match.group('amount')
    if not amount or amount == '.':
        raise exceptions.InvalidArgumentException('--cpu', 'Invalid amount for: {} specified.'.format(cpu))
    if suffix and suffix not in _GCF_GEN2_CPU_UNITS:
        raise exceptions.InvalidArgumentException('--cpu', 'Invalid suffix for: {} specified.'.format(cpu))
    parsed_memory = amount + suffix
    return parsed_memory