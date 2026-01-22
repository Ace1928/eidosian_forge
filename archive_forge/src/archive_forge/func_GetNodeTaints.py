from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetNodeTaints(args):
    """Gets node taint objects from the arguments.

  Args:
    args: Arguments parsed from the command.

  Returns:
    The list of node taint objects.

  Raises:
    ArgumentError: If the node taint format is invalid.
  """
    taints = []
    taint_effect_map = {_ToCamelCase(e): e for e in _TAINT_EFFECT_ENUM_MAPPER.choices}
    node_taints = getattr(args, 'node_taints', None)
    if node_taints:
        for k, v in node_taints.items():
            value, effect = _ValidateNodeTaintFormat(v)
            effect = taint_effect_map[effect]
            effect = _TAINT_EFFECT_ENUM_MAPPER.GetEnumForChoice(effect)
            taint = api_util.GetMessagesModule().GoogleCloudGkemulticloudV1NodeTaint(key=k, value=value, effect=effect)
            taints.append(taint)
    return taints