from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def _ArgNameMatches(self, name, other_name):
    """Checks if two argument names match in the namespace.

    RESOURCE_ARG and --resource-arg will match with each other, as well as exact
    matches.

    Args:
      name: the first argument name.
      other_name: the second argument name.

    Returns:
      (bool) True if the names match.
    """
    if util.NormalizeFormat(name) == util.NormalizeFormat(other_name):
        return True
    return False