from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateExecStepConfig(messages, arg_name, path, allowed_success_codes, is_windows):
    """Creates an ExecStepConfig message from input arguments."""
    interpreter = messages.ExecStepConfig.InterpreterValueValuesEnum.INTERPRETER_UNSPECIFIED
    gcs_params = osconfig_command_utils.GetGcsParams(arg_name, path)
    if gcs_params:
        if is_windows:
            interpreter = _GetWindowsExecStepConfigInterpreter(messages, gcs_params['object'])
        return messages.ExecStepConfig(gcsObject=messages.GcsObject(bucket=gcs_params['bucket'], object=gcs_params['object'], generationNumber=gcs_params['generationNumber']), allowedSuccessCodes=allowed_success_codes if allowed_success_codes else [], interpreter=interpreter)
    else:
        if is_windows:
            interpreter = _GetWindowsExecStepConfigInterpreter(messages, path)
        return messages.ExecStepConfig(localPath=path, allowedSuccessCodes=allowed_success_codes if allowed_success_codes else [], interpreter=interpreter)