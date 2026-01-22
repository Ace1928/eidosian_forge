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
class BFDSession(object):
    """BFD Session class.

    An instance maintains a BFD session.
    """

    def __init__(self, app, my_discr, dpid, ofport, src_mac, src_ip, src_port, dst_mac='FF:FF:FF:FF:FF:FF', dst_ip='255.255.255.255', detect_mult=3, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, auth_type=0, auth_keys=None):
        """
        Initialize a BFD session.

        __init__ takes the corresponding args in this order.

        .. tabularcolumns:: |l|L|

        ========================= ============================================
        Argument                  Description
        ========================= ============================================
        app                       The instance of BFDLib.
        my_discr                  My Discriminator.
        dpid                      Datapath ID of the BFD interface.
        ofport                    Openflow port number of the BFD interface.
        src_mac                   Source MAC address of the BFD interface.
        src_ip                    Source IPv4 address of the BFD interface.
        dst_mac                   (Optional) Destination MAC address of the
                                  BFD interface.
        dst_ip                    (Optional) Destination IPv4 address of the
                                  BFD interface.
        detect_mult               (Optional) Detection time multiplier.
        desired_min_tx_interval   (Optional) Desired Min TX Interval.
                                  (in microseconds)
        required_min_rx_interval  (Optional) Required Min RX Interval.
                                  (in microseconds)
        auth_type                 (Optional) Authentication type.
        auth_keys                 (Optional) A dictionary of authentication
                                  key chain which key is an integer of
                                  *Auth Key ID* and value is a string of
                                  *Password* or *Auth Key*.
        ========================= ============================================

        Example::

            sess = BFDSession(app=self.bfdlib,
                              my_discr=1,
                              dpid=1,
                              ofport=1,
                              src_mac="01:23:45:67:89:AB",
                              src_ip="192.168.1.1",
                              dst_mac="12:34:56:78:9A:BC",
                              dst_ip="192.168.1.2",
                              detect_mult=3,
                              desired_min_tx_interval=1000000,
                              required_min_rx_interval=1000000,
                              auth_type=bfd.BFD_AUTH_KEYED_SHA1,
                              auth_keys={1: "secret key 1",
                                         2: "secret key 2"})
        """
        auth_keys = auth_keys if auth_keys else {}
        assert not (auth_type and len(auth_keys) == 0)
        self.app = app
        self._session_state = bfd.BFD_STATE_DOWN
        self._remote_session_state = bfd.BFD_STATE_DOWN
        self._local_discr = my_discr
        self._remote_discr = 0
        self._local_diag = 0
        self._desired_min_tx_interval = 1000000
        self._required_min_rx_interval = required_min_rx_interval
        self._remote_min_rx_interval = -1
        self._demand_mode = 0
        self._remote_demand_mode = 0
        self._detect_mult = detect_mult
        self._auth_type = auth_type
        self._auth_keys = auth_keys
        if self._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
            self._rcv_auth_seq = 0
            self._xmit_auth_seq = random.randint(0, UINT32_MAX)
            self._auth_seq_known = 0
        self._cfg_desired_min_tx_interval = desired_min_tx_interval
        self._cfg_required_min_echo_rx_interval = 0
        self._active_role = True
        self._detect_time = 0
        self._xmit_period = None
        self._update_xmit_period()
        self._is_polling = True
        self._pending_final = False
        self._enable_send = True
        self._lock = None
        self.src_mac = src_mac
        self.dst_mac = dst_mac
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.ipv4_id = random.randint(0, UINT16_MAX)
        self.src_port = src_port
        self.dst_port = BFD_CONTROL_UDP_PORT
        if dst_mac == 'FF:FF:FF:FF:FF:FF' or dst_ip == '255.255.255.255':
            self._remote_addr_config = False
        else:
            self._remote_addr_config = True
        self.dpid = dpid
        self.datapath = None
        self.ofport = ofport
        hub.spawn(self._send_loop)
        LOG.info('[BFD][%s][INIT] BFD Session initialized.', hex(self._local_discr))

    @property
    def my_discr(self):
        """
        Returns My Discriminator of the BFD session.
        """
        return self._local_discr

    @property
    def your_discr(self):
        """
        Returns Your Discriminator of the BFD session.
        """
        return self._remote_discr

    def set_remote_addr(self, dst_mac, dst_ip):
        """
        Configure remote ethernet and IP addresses.
        """
        self.dst_mac = dst_mac
        self.dst_ip = dst_ip
        if not (dst_mac == 'FF:FF:FF:FF:FF:FF' or dst_ip == '255.255.255.255'):
            self._remote_addr_config = True
        LOG.info('[BFD][%s][REMOTE] Remote address configured: %s, %s.', hex(self._local_discr), self.dst_ip, self.dst_mac)

    def recv(self, bfd_pkt):
        """
        BFD packet receiver.
        """
        LOG.debug('[BFD][%s][RECV] BFD Control received: %s', hex(self._local_discr), bytes(bfd_pkt))
        self._remote_discr = bfd_pkt.my_discr
        self._remote_state = bfd_pkt.state
        self._remote_demand_mode = bfd_pkt.flags & bfd.BFD_FLAG_DEMAND
        if self._remote_min_rx_interval != bfd_pkt.required_min_rx_interval:
            self._remote_min_rx_interval = bfd_pkt.required_min_rx_interval
            self._update_xmit_period()
        if bfd_pkt.flags & bfd.BFD_FLAG_FINAL and self._is_polling:
            self._is_polling = False
        if self._session_state == bfd.BFD_STATE_ADMIN_DOWN:
            return
        if bfd_pkt.state == bfd.BFD_STATE_ADMIN_DOWN:
            if self._session_state != bfd.BFD_STATE_DOWN:
                self._set_state(bfd.BFD_STATE_DOWN, bfd.BFD_DIAG_NEIG_SIG_SESS_DOWN)
        elif self._session_state == bfd.BFD_STATE_DOWN:
            if bfd_pkt.state == bfd.BFD_STATE_DOWN:
                self._set_state(bfd.BFD_STATE_INIT)
            elif bfd_pkt.state == bfd.BFD_STATE_INIT:
                self._set_state(bfd.BFD_STATE_UP)
        elif self._session_state == bfd.BFD_STATE_INIT:
            if bfd_pkt.state in [bfd.BFD_STATE_INIT, bfd.BFD_STATE_UP]:
                self._set_state(bfd.BFD_STATE_UP)
        elif bfd_pkt.state == bfd.BFD_STATE_DOWN:
            self._set_state(bfd.BFD_STATE_DOWN, bfd.BFD_DIAG_NEIG_SIG_SESS_DOWN)
        if self._remote_demand_mode and self._session_state == bfd.BFD_STATE_UP and (self._remote_session_state == bfd.BFD_STATE_UP):
            self._enable_send = False
        if not self._remote_demand_mode or self._session_state != bfd.BFD_STATE_UP or self._remote_session_state != bfd.BFD_STATE_UP:
            if not self._enable_send:
                self._enable_send = True
                hub.spawn(self._send_loop)
        if self._detect_time == 0:
            self._detect_time = bfd_pkt.desired_min_tx_interval * bfd_pkt.detect_mult / 1000000.0
            hub.spawn(self._recv_timeout_loop)
        if bfd_pkt.flags & bfd.BFD_FLAG_POLL:
            self._pending_final = True
            self._detect_time = bfd_pkt.desired_min_tx_interval * bfd_pkt.detect_mult / 1000000.0
        if self._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
            self._rcv_auth_seq = bfd_pkt.auth_cls.seq
            self._auth_seq_known = 1
        if self._lock is not None:
            self._lock.set()

    def _set_state(self, new_state, diag=None):
        """
        Set the state of the BFD session.
        """
        old_state = self._session_state
        LOG.info('[BFD][%s][STATE] State changed from %s to %s.', hex(self._local_discr), bfd.BFD_STATE_NAME[old_state], bfd.BFD_STATE_NAME[new_state])
        self._session_state = new_state
        if new_state == bfd.BFD_STATE_DOWN:
            if diag is not None:
                self._local_diag = diag
            self._desired_min_tx_interval = 1000000
            self._is_polling = True
            self._update_xmit_period()
        elif new_state == bfd.BFD_STATE_UP:
            self._desired_min_tx_interval = self._cfg_desired_min_tx_interval
            self._is_polling = True
            self._update_xmit_period()
        self.app.send_event_to_observers(EventBFDSessionStateChanged(self, old_state, new_state))

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

    def _update_xmit_period(self):
        """
        Update transmission period of the BFD session.
        """
        if self._desired_min_tx_interval > self._remote_min_rx_interval:
            xmit_period = self._desired_min_tx_interval
        else:
            xmit_period = self._remote_min_rx_interval
        if self._detect_mult == 1:
            xmit_period *= random.randint(75, 90) / 100.0
        else:
            xmit_period *= random.randint(75, 100) / 100.0
        self._xmit_period = xmit_period / 1000000.0
        LOG.info('[BFD][%s][XMIT] Transmission period changed to %f', hex(self._local_discr), self._xmit_period)

    def _send_loop(self):
        """
        A loop to proceed periodic BFD packet transmission.
        """
        while self._enable_send:
            hub.sleep(self._xmit_period)
            if self._remote_discr == 0 and (not self._active_role):
                continue
            if self._remote_min_rx_interval == 0:
                continue
            if self._remote_demand_mode and self._session_state == bfd.BFD_STATE_UP and (self._remote_session_state == bfd.BFD_STATE_UP) and (not self._is_polling):
                continue
            self._send()

    def _send(self):
        """
        BFD packet sender.
        """
        if self.datapath is None:
            return
        flags = 0
        if self._pending_final:
            flags |= bfd.BFD_FLAG_FINAL
            self._pending_final = False
            self._is_polling = False
        if self._is_polling:
            flags |= bfd.BFD_FLAG_POLL
        auth_cls = None
        if self._auth_type:
            auth_key_id = list(self._auth_keys.keys())[random.randint(0, len(list(self._auth_keys.keys())) - 1)]
            auth_key = self._auth_keys[auth_key_id]
            if self._auth_type == bfd.BFD_AUTH_SIMPLE_PASS:
                auth_cls = bfd.SimplePassword(auth_key_id=auth_key_id, password=auth_key)
            if self._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_METICULOUS_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1, bfd.BFD_AUTH_METICULOUS_KEYED_SHA1]:
                if self._auth_type in [bfd.BFD_AUTH_KEYED_MD5, bfd.BFD_AUTH_KEYED_SHA1]:
                    if random.randint(0, 1):
                        self._xmit_auth_seq = self._xmit_auth_seq + 1 & UINT32_MAX
                else:
                    self._xmit_auth_seq = self._xmit_auth_seq + 1 & UINT32_MAX
                auth_cls = bfd.bfd._auth_parsers[self._auth_type](auth_key_id=auth_key_id, seq=self._xmit_auth_seq, auth_key=auth_key)
        if auth_cls is not None:
            flags |= bfd.BFD_FLAG_AUTH_PRESENT
        if self._demand_mode and self._session_state == bfd.BFD_STATE_UP and (self._remote_session_state == bfd.BFD_STATE_UP):
            flags |= bfd.BFD_FLAG_DEMAND
        diag = self._local_diag
        state = self._session_state
        detect_mult = self._detect_mult
        my_discr = self._local_discr
        your_discr = self._remote_discr
        desired_min_tx_interval = self._desired_min_tx_interval
        required_min_rx_interval = self._required_min_rx_interval
        required_min_echo_rx_interval = self._cfg_required_min_echo_rx_interval
        src_mac = self.src_mac
        dst_mac = self.dst_mac
        src_ip = self.src_ip
        dst_ip = self.dst_ip
        self.ipv4_id = self.ipv4_id + 1 & UINT16_MAX
        ipv4_id = self.ipv4_id
        src_port = self.src_port
        dst_port = self.dst_port
        data = BFDPacket.bfd_packet(src_mac=src_mac, dst_mac=dst_mac, src_ip=src_ip, dst_ip=dst_ip, ipv4_id=ipv4_id, src_port=src_port, dst_port=dst_port, diag=diag, state=state, flags=flags, detect_mult=detect_mult, my_discr=my_discr, your_discr=your_discr, desired_min_tx_interval=desired_min_tx_interval, required_min_rx_interval=required_min_rx_interval, required_min_echo_rx_interval=required_min_echo_rx_interval, auth_cls=auth_cls)
        datapath = self.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(self.ofport)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER, actions=actions, data=data)
        datapath.send_msg(out)
        LOG.debug('[BFD][%s][SEND] BFD Control sent.', hex(self._local_discr))