import logging
import socket
import traceback
import msgpack
from os_ken.lib.packet import safi as subaddr_family
from os_ken.services.protocols.bgp import api
from os_ken.services.protocols.bgp.api.base import ApiException
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import ORIGIN_RD
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import VPN_LABEL
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import FlexinetPeer
from os_ken.services.protocols.bgp.base import NET_CTRL_ERROR_CODE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
def _create_prefix_notification(outgoing_msg, rpc_session):
    """Constructs prefix notification with data from given outgoing message.

    Given RPC session is used to create RPC notification message.
    """
    assert outgoing_msg
    path = outgoing_msg.path
    assert path
    vpn_nlri = path.nlri
    assert path.source is not None
    if path.source != VRF_TABLE:
        params = [{ROUTE_DISTINGUISHER: outgoing_msg.route_dist, PREFIX: vpn_nlri.prefix, NEXT_HOP: path.nexthop, VRF_RF: VrfConf.rf_2_vrf_rf(path.route_family)}]
        if path.nlri.ROUTE_FAMILY.safi not in (subaddr_family.IP_FLOWSPEC, subaddr_family.VPN_FLOWSPEC):
            params[VPN_LABEL] = path.label_list[0]
        if not path.is_withdraw:
            rpc_msg = rpc_session.create_notification(NOTIFICATION_ADD_REMOTE_PREFIX, params)
        else:
            rpc_msg = rpc_session.create_notification(NOTIFICATION_DELETE_REMOTE_PREFIX, params)
    else:
        params = [{ROUTE_DISTINGUISHER: outgoing_msg.route_dist, PREFIX: vpn_nlri.prefix, NEXT_HOP: path.nexthop, VRF_RF: VrfConf.rf_2_vrf_rf(path.route_family), ORIGIN_RD: path.origin_rd}]
        if not path.is_withdraw:
            rpc_msg = rpc_session.create_notification(NOTIFICATION_ADD_LOCAL_PREFIX, params)
        else:
            rpc_msg = rpc_session.create_notification(NOTIFICATION_DELETE_LOCAL_PREFIX, params)
    return rpc_msg