import os
import socket
import struct
from os_ken import cfg
from os_ken.base.app_manager import OSKenApp
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.lib.packet import safi as packet_safi
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client import event as zclient_event
def _send_ip_route_impl(self, prefix, nexthops=None, safi=packet_safi.UNICAST, flags=zebra.ZEBRA_FLAG_INTERNAL, distance=None, metric=None, mtu=None, tag=None, is_withdraw=False):
    if ip.valid_ipv4(prefix):
        if is_withdraw:
            msg_cls = zebra.ZebraIPv4RouteDelete
        else:
            msg_cls = zebra.ZebraIPv4RouteAdd
    elif ip.valid_ipv6(prefix):
        if is_withdraw:
            msg_cls = zebra.ZebraIPv6RouteDelete
        else:
            msg_cls = zebra.ZebraIPv6RouteAdd
    else:
        raise ValueError('Invalid prefix: %s' % prefix)
    nexthop_list = []
    for nexthop in nexthops:
        if ip.valid_ipv4(nexthop):
            nexthop_list.append(zebra.NextHopIPv4(addr=nexthop))
        elif ip.valid_ipv6(nexthop):
            nexthop_list.append(zebra.NextHopIPv6(addr=nexthop))
        else:
            raise ValueError('Invalid nexthop: %s' % nexthop)
    msg = zebra.ZebraMessage(version=self.zserv_ver, body=msg_cls(route_type=self.route_type, flags=flags, message=0, safi=safi, prefix=prefix, nexthops=nexthop_list, distance=distance, metric=metric, mtu=mtu, tag=tag, instance=0))
    self.send_msg(msg)
    return msg