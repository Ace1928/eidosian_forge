from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib import init_util
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import network_diagnostics
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _CreateBotoConfig(self):
    gsutil_path = _FindGsutil()
    if not gsutil_path:
        log.debug('Unable to find [gsutil]. Not configuring default .boto file')
        return
    boto_path = files.ExpandHomeDir(os.path.join('~', '.boto'))
    if os.path.exists(boto_path):
        log.debug('Not configuring default .boto file. File already exists at [{boto_path}].'.format(boto_path=boto_path))
        return
    command_args = ['config', '-n', '-o', boto_path]
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        gsutil_args = execution_utils.ArgsForCMDTool(gsutil_path, *command_args)
    else:
        gsutil_args = execution_utils.ArgsForExecutableTool(gsutil_path, *command_args)
    return_code = execution_utils.Exec(gsutil_args, no_exit=True, out_func=log.file_only_logger.debug, err_func=log.file_only_logger.debug)
    if return_code == 0:
        log.status.write('Created a default .boto configuration file at [{boto_path}]. See this file and\n[https://cloud.google.com/storage/docs/gsutil/commands/config] for more\ninformation about configuring Google Cloud Storage.\n'.format(boto_path=boto_path))
    else:
        log.status.write('Error creating a default .boto configuration file. Please run [gsutil config -n] if you would like to create this file.\n')