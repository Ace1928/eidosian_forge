from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import socket
from apitools.base.py import encoding
from googlecloudsdk.api_lib.runtime_config import exceptions as rtc_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as sdk_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def ParseVariableName(variable_name, args):
    """Parse a variable name or URL, and return a resource.

  Args:
    variable_name: The variable name.
    args: CLI arguments, possibly containing a config name.

  Returns:
    The parsed resource.
  """
    params = {'projectsId': lambda: ParseConfigName(ConfigName(args)).projectsId, 'configsId': lambda: ParseConfigName(ConfigName(args)).configsId}
    return resources.REGISTRY.Parse(variable_name, collection='runtimeconfig.projects.configs.variables', params=params)