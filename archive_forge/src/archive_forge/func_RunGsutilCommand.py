from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import re
import string
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def RunGsutilCommand(command_name, command_args=None, run_concurrent=False, out_func=log.file_only_logger.debug, err_func=log.file_only_logger.debug):
    """Runs the specified gsutil command and returns the command's exit code.

  WARNING: This is not compatible with python 3 and should no longer be used.

  Args:
    command_name: The gsutil command to run.
    command_args: List of arguments to pass to the command.
    run_concurrent: Whether concurrent uploads should be enabled while running
      the command.
    out_func: str->None, a function to call with the stdout of the gsutil
        command.
    err_func: str->None, a function to call with the stderr of the gsutil
        command.

  Returns:
    The exit code of the call to the gsutil command.
  """
    command_path = _GetGsutilPath()
    args = ['-m', command_name] if run_concurrent else [command_name]
    if command_args is not None:
        args += command_args
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        gsutil_args = execution_utils.ArgsForCMDTool(command_path + '.cmd', *args)
    else:
        gsutil_args = execution_utils.ArgsForExecutableTool(command_path, *args)
    log.debug('Running command: [{args}]]'.format(args=' '.join(gsutil_args)))
    return execution_utils.Exec(gsutil_args, no_exit=True, out_func=out_func, err_func=err_func)