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
def _FindGsutil():
    """Finds the bundled gsutil wrapper.

  Returns:
    The path to gsutil.
  """
    sdk_bin_path = config.Paths().sdk_bin_path
    if not sdk_bin_path:
        return
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        gsutil = 'gsutil.cmd'
    else:
        gsutil = 'gsutil'
    return os.path.join(sdk_bin_path, gsutil)