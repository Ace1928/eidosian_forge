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
def _RunCmd(self, cmd, params=None, disable_user_output=True):
    if not self._cli_power_users_only.IsValidCommand(cmd):
        log.info('Command %s does not exist.', cmd)
        return None
    if params is None:
        params = []
    args = cmd + params
    log.info('Executing: [gcloud %s]', ' '.join(args))
    try:
        if disable_user_output:
            args.append('--no-user-output-enabled')
        if properties.VALUES.core.verbosity.Get() is None and disable_user_output:
            args.append('--verbosity=none')
        if properties.VALUES.core.log_http.GetBool():
            args.append('--log-http')
        return resource_projector.MakeSerializable(self.ExecuteCommandDoNotUse(args))
    except SystemExit as exc:
        log.info('[%s] has failed\n', ' '.join(cmd + params))
        raise c_exc.FailedSubCommand(cmd + params, exc.code)
    except BaseException:
        log.info('Failed to run [%s]\n', ' '.join(cmd + params))
        raise