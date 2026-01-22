from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
def _ValidateSupportedOsType(short_name, version):
    """Validates the combination of the OS short name and version is supported.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is an OpsAgentPolicy.Assignment.OsType object. Also the
  OS short name has been already validated at the arg parsing stage.

  Args:
    short_name: str. The OS short name to filter instances by.
    version: str. The OS version to filter instances by.

  Returns:
    An empty list if the validation passes. A singleton list with the following
    error if the validation fails.
    * OsTypeNotSupportedError:
      The combination of the OS short name and version is not supported.
  """
    if short_name in _SUPPORTED_OS_SHORT_NAMES_AND_VERSIONS and short_name not in _OS_SHORT_NAMES_WITH_OS_AGENT_PREINSTALLED:
        log.warning('For the policies to take effect on [{}] OS distro, please follow the instructions at https://cloud.google.com/compute/docs/manage-os#agent-install to install the OS Config Agent on your instances.'.format(short_name))
    supported_versions = _SUPPORTED_OS_SHORT_NAMES_AND_VERSIONS[short_name]
    if any((version.startswith(v) for v in supported_versions)):
        return []
    return [OsTypeNotSupportedError(short_name, version)]