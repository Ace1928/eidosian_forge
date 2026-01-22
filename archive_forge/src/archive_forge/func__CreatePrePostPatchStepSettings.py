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
def _CreatePrePostPatchStepSettings(args, messages, is_pre_patch_step):
    """Creates an ExecStep message from input arguments."""
    if is_pre_patch_step:
        if not any([args.pre_patch_linux_executable, args.pre_patch_linux_success_codes, args.pre_patch_windows_executable, args.pre_patch_windows_success_codes]):
            return None
        _ValidatePrePostPatchStepArgs('pre-patch-linux-executable', args.pre_patch_linux_executable, 'pre-patch-linux-success-codes', args.pre_patch_linux_success_codes)
        _ValidatePrePostPatchStepArgs('pre-patch-windows-executable', args.pre_patch_windows_executable, 'pre-patch-windows-success-codes', args.pre_patch_windows_success_codes)
        pre_patch_linux_step_config = pre_patch_windows_step_config = None
        if args.pre_patch_linux_executable:
            pre_patch_linux_step_config = _CreateExecStepConfig(messages, 'pre-patch-linux-executable', args.pre_patch_linux_executable, args.pre_patch_linux_success_codes, is_windows=False)
        if args.pre_patch_windows_executable:
            pre_patch_windows_step_config = _CreateExecStepConfig(messages, 'pre-patch-windows-executable', args.pre_patch_windows_executable, args.pre_patch_windows_success_codes, is_windows=True)
        return messages.ExecStep(linuxExecStepConfig=pre_patch_linux_step_config, windowsExecStepConfig=pre_patch_windows_step_config)
    else:
        if not any([args.post_patch_linux_executable, args.post_patch_linux_success_codes, args.post_patch_windows_executable, args.post_patch_windows_success_codes]):
            return None
        _ValidatePrePostPatchStepArgs('post-patch-linux-executable', args.post_patch_linux_executable, 'post-patch-linux-success-codes', args.post_patch_linux_success_codes)
        _ValidatePrePostPatchStepArgs('post-patch-windows-executable', args.post_patch_windows_executable, 'post-patch-windows-success-codes', args.post_patch_windows_success_codes)
        post_patch_linux_step_config = post_patch_windows_step_config = None
        if args.post_patch_linux_executable:
            post_patch_linux_step_config = _CreateExecStepConfig(messages, 'post-patch-linux-executable', args.post_patch_linux_executable, args.post_patch_linux_success_codes, is_windows=False)
        if args.post_patch_windows_executable:
            post_patch_windows_step_config = _CreateExecStepConfig(messages, 'post-patch-windows-executable', args.post_patch_windows_executable, args.post_patch_windows_success_codes, is_windows=True)
        return messages.ExecStep(linuxExecStepConfig=post_patch_linux_step_config, windowsExecStepConfig=post_patch_windows_step_config)