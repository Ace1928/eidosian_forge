from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
def GetDest(self, parameter_name, prefix=None):
    """Returns the argument parser dest name for parameter_name with prefix.

    Args:
      parameter_name: The resource parameter name.
      prefix: The prefix name for parameter_name if not None.

    Returns:
      The argument parser dest name for parameter_name.
    """
    del prefix
    attribute_name = self._AttributeName(parameter_name)
    flag_name = self.resource_info.attribute_to_args_map.get(attribute_name, None)
    if not flag_name:
        return None
    return util.NamespaceFormat(flag_name)