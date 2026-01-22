from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBPeerConfigs(bgp_lb_config_group, is_update=False):
    """Adds flags for peer configs used by BGP LB load balancer.

  Args:
    bgp_lb_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    bgp_lb_peer_configs_mutex_group = bgp_lb_config_group.add_group(help='BGP LB peer configuration.', mutex=True, required=required)
    bgp_lb_peer_configs_from_file_help_text = '\nPath of the YAML/JSON file that contains the BGP LB peer configs.\n\nExamples:\n\n  bgpPeerConfigs:\n  - asn: 1000\n    controlPlaneNodes:\n    - 10.200.0.14/32\n    - 10.200.0.15/32\n    ipAddress: 10.200.0.16/32\n  - asn: 1001\n    controlPlaneNodes:\n    - 10.200.0.17/32\n    - 10.200.0.18/32\n    ipAddress: 10.200.0.19/32\n\nList of supported fields in `bgpPeerConfigs`\n\nKEY               | VALUE                 | NOTE\n------------------|-----------------------|---------------------------\nasn               | int                   | required, mutable\ncontrolPlaneNodes | one or more IP ranges | optional, mutable\nipAddress         | valid IP address      | required, mutable\n\n'
    bgp_lb_peer_configs_mutex_group.add_argument('--bgp-lb-peer-configs-from-file', help=bgp_lb_peer_configs_from_file_help_text, type=arg_parsers.YAMLFileContents(), hidden=True)
    bgp_lb_peer_configs_mutex_group.add_argument('--bgp-lb-peer-configs', help='BGP LB peer configuration.', action='append', type=arg_parsers.ArgDict(spec={'asn': int, 'ip-address': str, 'control-plane-nodes': arg_parsers.ArgList(custom_delim_char=';')}, required_keys=['asn', 'ip-address']))