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
def _GetRetry(args: parser_extensions.Namespace, messages: types.ModuleType, event_trigger: Optional[api_types.EventTrigger]) -> Tuple[api_types.RetryPolicy, FrozenSet[str]]:
    """Constructs an RetryPolicy enum from --(no-)retry flag.

  Args:
    args: arguments that this command was invoked with.
    messages: messages module, the GCFv2 message stubs.
    event_trigger: trigger used to request events sent from another service.

  Returns:
    A tuple `(retry_policy, update_fields_set)` where:
    - `retry_policy` is the retry policy enum value,
    - `update_fields_set` is the set of update mask fields.
  """
    if event_trigger is None:
        raise exceptions.FunctionsError(_INVALID_RETRY_FLAG_ERROR_MESSAGE)
    if args.retry:
        return (messages.EventTrigger.RetryPolicyValueValuesEnum('RETRY_POLICY_RETRY'), frozenset(['eventTrigger.retryPolicy']))
    else:
        return (messages.EventTrigger.RetryPolicyValueValuesEnum('RETRY_POLICY_DO_NOT_RETRY'), frozenset(['eventTrigger.retryPolicy']))