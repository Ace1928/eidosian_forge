from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddBGPNodeConfigs(bgp_lb_config_group, is_update=False):
    """Adds a flag for BGP node config fields.

  Args:
    bgp_lb_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    bgp_node_configs_help_text = "\nBGP load balancer data plane node configurations.\n\nExamples:\n\nTo specify configurations for two nodes of IP `192.168.0.1` and `192.168.1.1`,\n\n```\n$ {command} example_cluster\n      --bgp-load-balancer-node-configs 'node-ip=192.168.0.1,labels=KEY1=VALUE1;KEY2=VALUE2'\n      --bgp-load-balancer-node-configs 'node-ip=192.168.1.1,labels=KEY3=VALUE3'\n```\n\nUse quote around the flag value to escape semicolon in the terminal.\n"
    required = not is_update
    bgp_lb_config_group.add_argument('--bgp-load-balancer-node-configs', help=bgp_node_configs_help_text, required=required, metavar='node-ip=IP,labels=KEY1=VALUE1;KEY2=VALUE2', action='append', type=arg_parsers.ArgDict(spec={'node-ip': str, 'labels': str}, required_keys=['node-ip']))