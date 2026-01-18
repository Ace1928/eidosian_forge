import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
Represents a way of reaching an VPNv6 Flow Specification destination.