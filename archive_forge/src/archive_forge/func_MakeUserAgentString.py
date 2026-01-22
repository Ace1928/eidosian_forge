from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def MakeUserAgentString(cmd_path=None):
    """Return a user-agent string for this request.

  Contains 'gcloud' in addition to several other product IDs used for tracing in
  metrics reporting.

  Args:
    cmd_path: str representing the current command for tracing.

  Returns:
    str, User Agent string.
  """
    user_platform = platforms.Platform.Current()
    architecture = GetAndCacheArchitecture(user_platform)
    return 'gcloud/{version} command/{cmd} invocation-id/{inv_id} environment/{environment} environment-version/{env_version} client-os/{os} client-os-ver/{os_version} client-pltf-arch/{architecture} interactive/{is_interactive} from-script/{from_script} python/{py_version} term/{term} {ua_fragment}'.format(version=config.CLOUD_SDK_VERSION.replace(' ', '_'), cmd=cmd_path or properties.VALUES.metrics.command_name.Get(), inv_id=INVOCATION_ID, environment=properties.GetMetricsEnvironment(), env_version=properties.VALUES.metrics.environment_version.Get(), os=user_platform.operating_system, os_version=user_platform.operating_system.clean_version if user_platform.operating_system else None, architecture=architecture, is_interactive=console_io.IsInteractive(error=True, heuristic=True), py_version=platform.python_version(), ua_fragment=user_platform.UserAgentFragment(), from_script=console_io.IsRunFromShellScript(), term=console_attr.GetConsoleAttr().GetTermIdentifier())