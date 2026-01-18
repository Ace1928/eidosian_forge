import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
Represents a way of reaching an VPNv4 Flow Specification destination.