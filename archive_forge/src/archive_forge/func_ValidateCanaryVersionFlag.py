from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ValidateCanaryVersionFlag(flag_name, version_map):
    """Retrieves canary version from input map.

  Args:
    flag_name: name of the flag associated with the parsed string.
    version_map: map containing version data provided by the user.
  """
    if version_map and TARGET_SIZE_NAME not in version_map:
        raise exceptions.RequiredArgumentException('{} {}={}'.format(flag_name, TARGET_SIZE_NAME, TARGET_SIZE_NAME.upper()), 'target size must be specified for canary version')