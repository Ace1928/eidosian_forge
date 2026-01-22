from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def _ValidateNodeTaintFormat(taint):
    """Validates the node taint format.

  Node taint is of format key=value:effect.

  Args:
    taint: Node taint.

  Returns:
    The node taint value and effect if the format is valid.

  Raises:
    ArgumentError: If the node taint format is invalid.
  """
    strs = taint.split(':')
    if len(strs) != 2:
        raise _InvalidValueError(taint, '--node-taints', _TAINT_FORMAT_HELP)
    value, effect = (strs[0], strs[1])
    return (value, effect)