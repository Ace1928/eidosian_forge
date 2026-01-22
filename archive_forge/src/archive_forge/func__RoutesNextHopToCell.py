from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _RoutesNextHopToCell(route):
    """Returns the next hop value in a compact form."""
    if route.get('nextHopInstance'):
        return path_simplifier.ScopedSuffix(route.get('nextHopInstance'))
    elif route.get('nextHopGateway'):
        return path_simplifier.ScopedSuffix(route.get('nextHopGateway'))
    elif route.get('nextHopIp'):
        return route.get('nextHopIp')
    elif route.get('nextHopVpnTunnel'):
        return path_simplifier.ScopedSuffix(route.get('nextHopVpnTunnel'))
    elif route.get('nextHopPeering'):
        return route.get('nextHopPeering')
    else:
        return ''