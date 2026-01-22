from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.service_extensions import wasm_plugin_api
from googlecloudsdk.api_lib.service_extensions import wasm_plugin_version_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.service_extensions import flags
from googlecloudsdk.command_lib.service_extensions import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetLogConfig(args):
    """Converts the dict representation of the log_config to proto.

  Args:
    args: args with log_level parsed ordered dict. If log-level flag is set,
          enable option should also be set.

  Returns:
    a value of messages.WasmPluginLogConfig or None,
    if log-level flag were not provided.
  """
    if args.log_config is None:
        return None
    return util.GetLogConfig(args.log_config[0])