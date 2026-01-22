import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
class LinkState(dict):

    def __init__(self):
        super(LinkState, self).__init__()
        self._map = defaultdict(lambda: defaultdict(lambda: None))

    def get_peers(self, src):
        return self._map[src].keys()

    def update_link(self, src, dst):
        link = Link(src, dst)
        self[link] = time.time()
        self._map[src][dst] = link
        rev_link = Link(dst, src)
        return rev_link in self

    def link_down(self, link):
        del self[link]
        del self._map[link.src][link.dst]

    def rev_link_set_timestamp(self, rev_link, timestamp):
        if rev_link in self:
            self[rev_link] = timestamp

    def port_deleted(self, src):
        dsts = self.get_peers(src)
        rev_link_dsts = []
        for dst in dsts:
            link = Link(src, dst)
            rev_link = Link(dst, src)
            del self[link]
            self.pop(rev_link, None)
            if src in self._map[dst]:
                del self._map[dst][src]
                rev_link_dsts.append(dst)
        del self._map[src]
        return (dsts, rev_link_dsts)