from collections import OrderedDict
import logging
import unittest
from unittest import mock
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import EvpnArbitraryEsi
from os_ken.lib.packet.bgp import EvpnLACPEsi
from os_ken.lib.packet.bgp import EvpnEthernetAutoDiscoveryNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.services.protocols.bgp.bgpspeaker import EVPN_MAX_ET
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.core import BgpCoreError
from os_ken.services.protocols.bgp.core_managers import table_manager
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
def test_update_vrf_table_l2_evpn_with_esi_dict(self):
    route_dist = '65000:100'
    prefix_str = None
    kwargs = {'ethernet_tag_id': EVPN_MAX_ET}
    esi = EvpnLACPEsi(mac_addr='aa:bb:cc:dd:ee:ff', port_key=100)
    prefix_inst = EvpnEthernetAutoDiscoveryNLRI(route_dist=route_dist, esi=esi, **kwargs)
    next_hop = '0.0.0.0'
    route_family = VRF_RF_L2_EVPN
    route_type = EvpnEthernetAutoDiscoveryNLRI.ROUTE_TYPE_NAME
    kwargs['esi'] = {'type': ESI_TYPE_LACP, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'port_key': 100}
    self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)