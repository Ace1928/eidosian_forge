from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bgp_lb_load_balancer_node_pool_config(self, args: parser_extensions.Namespace):
    """Constructs proto message BareMetalNodePoolConfig."""
    if 'bgp_lb_load_balancer_node_configs_from_file' in args.GetSpecifiedArgsDict():
        bgp_lb_node_configs = self._bgp_lb_node_configs_from_file(args)
    else:
        bgp_lb_node_configs = self._bgp_lb_node_configs_from_flag(args)
    kwargs = {'nodeConfigs': bgp_lb_node_configs, 'labels': self._bgp_lb_labels(args), 'taints': self._bgp_lb_node_taints(args)}
    if any(kwargs.values()):
        return messages.BareMetalNodePoolConfig(**kwargs)
    return None