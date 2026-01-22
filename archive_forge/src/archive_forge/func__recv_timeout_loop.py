import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
def _recv_timeout_loop(self):
    """
        A loop to check timeout of receiving remote BFD packet.
        """
    while self._detect_time:
        last_wait = time.time()
        self._lock = hub.Event()
        self._lock.wait(timeout=self._detect_time)
        if self._lock.is_set():
            if getattr(self, '_auth_seq_known', 0):
                if last_wait > time.time() + 2 * self._detect_time:
                    self._auth_seq_known = 0
        else:
            LOG.info('[BFD][%s][RECV] BFD Session timed out.', hex(self._local_discr))
            if self._session_state not in [bfd.BFD_STATE_DOWN, bfd.BFD_STATE_ADMIN_DOWN]:
                self._set_state(bfd.BFD_STATE_DOWN, bfd.BFD_DIAG_CTRL_DETECT_TIME_EXPIRED)
            if getattr(self, '_auth_seq_known', 0):
                self._auth_seq_known = 0