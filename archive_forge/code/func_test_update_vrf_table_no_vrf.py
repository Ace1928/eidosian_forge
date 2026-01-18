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
@mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
def test_update_vrf_table_no_vrf(self):
    route_dist = '65000:100'
    ip_network = '192.168.0.0'
    ip_prefix_len = 24
    prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
    next_hop = '10.0.0.1'
    route_family = VRF_RF_IPV4
    route_type = None
    kwargs = {}
    tbl_mng = table_manager.TableCoreManager(None, None)
    tbl_mng._tables = {}
    self.assertRaises(BgpCoreError, tbl_mng.update_vrf_table, route_dist=route_dist, prefix=prefix_str, next_hop=next_hop, route_family=route_family, route_type=route_type, **kwargs)