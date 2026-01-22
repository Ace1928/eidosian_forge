from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _GraphvizNode(res_id: runapps.ResourceID, in_count: int, out_count: int) -> str:
    """Style for the node.

  Args:
    res_id: resource ID of the node
    in_count: number of bindings going into this node
    out_count: number of bindings coming out of this node

  Returns:
    node style code in DOT
  """
    ingress = in_count == 0 and out_count > 0
    backing = out_count == 0 and in_count > 0
    if ingress:
        color = '#e37400'
        fillcolor = '#fdd663'
    elif backing:
        color = '#0d652d'
        fillcolor = '#81c995'
    else:
        color = '#174ea6'
        fillcolor = '#8ab4f8'
    return '  "{}" [label="{}" color="{}" fillcolor="{}"]'.format(_NodeName(res_id), _NodeLabel(res_id), color, fillcolor)