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
def _GetLabels(args: parser_extensions.Namespace, messages: types.ModuleType, existing_function: Optional[api_types.Function]) -> Tuple[Optional[api_types.LabelsValue], FrozenSet[str]]:
    """Constructs labels from command-line arguments.

  Args:
    args: The arguments that this command was invoked with
    messages: messages module, the GCFv2 message stubs.
    existing_function: The pre-existing function.

  Returns:
    A tuple `(labels, updated_fields_set)` where:
    - `labels` is functions labels metadata,
    - `updated_fields_set` is the set of update mask fields.
  """
    if existing_function:
        required_labels = {}
    else:
        required_labels = {_DEPLOYMENT_TOOL_LABEL: _DEPLOYMENT_TOOL_VALUE}
    labels_diff = labels_util.Diff.FromUpdateArgs(args, required_labels=required_labels)
    labels_update = labels_diff.Apply(messages.Function.LabelsValue, existing_function.labels if existing_function else None)
    if labels_update.needs_update:
        return (labels_update.labels, frozenset(['labels']))
    else:
        return (None, frozenset())