import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def _transmit_bpdu(self):
    while True:
        if self.role == DESIGNATED_PORT:
            now = datetime.datetime.today()
            if self.send_tc_timer and self.send_tc_timer < now:
                self.send_tc_timer = None
                self.send_tc_flg = False
            if not self.send_tc_flg:
                flags = 0
                log_msg = '[port=%d] Send Config BPDU.'
            else:
                flags = 1
                log_msg = '[port=%d] Send TopologyChange BPDU.'
            bpdu_data = self._generate_config_bpdu(flags)
            self.ofctl.send_packet_out(self.ofport.port_no, bpdu_data)
            self.logger.debug(log_msg, self.ofport.port_no, extra=self.dpid_str)
        if self.send_tcn_flg:
            bpdu_data = self._generate_tcn_bpdu()
            self.ofctl.send_packet_out(self.ofport.port_no, bpdu_data)
            self.logger.debug('[port=%d] Send TopologyChangeNotify BPDU.', self.ofport.port_no, extra=self.dpid_str)
        hub.sleep(self.port_times.hello_time)