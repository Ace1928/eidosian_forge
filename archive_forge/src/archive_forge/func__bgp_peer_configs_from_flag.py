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
def _bgp_peer_configs_from_flag(self, args: parser_extensions.Namespace):
    if not args.bgp_lb_peer_configs:
        return []
    peer_configs = []
    for peer_config in args.bgp_lb_peer_configs:
        peer_configs.append(messages.BareMetalStandaloneBgpPeerConfig(controlPlaneNodes=peer_config.get('control-plane-nodes', []), asn=peer_config.get('asn', None), ipAddress=peer_config.get('ip-address', None)))
    return peer_configs