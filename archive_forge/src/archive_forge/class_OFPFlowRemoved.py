import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
@_register_parser
@_set_msg_type(ofproto.OFPT_FLOW_REMOVED)
class OFPFlowRemoved(MsgBase):
    """
    Flow removed message

    When flow entries time out or are deleted, the switch notifies controller
    with this message.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    cookie           Opaque controller-issued identifier
    priority         Priority level of flow entry
    reason           One of the following values.

                     | OFPRR_IDLE_TIMEOUT
                     | OFPRR_HARD_TIMEOUT
                     | OFPRR_DELETE
                     | OFPRR_GROUP_DELETE
    table_id         ID of the table
    duration_sec     Time flow was alive in seconds
    duration_nsec    Time flow was alive in nanoseconds beyond duration_sec
    idle_timeout     Idle timeout from original flow mod
    hard_timeout     Hard timeout from original flow mod
    packet_count     Number of packets that was associated with the flow
    byte_count       Number of bytes that was associated with the flow
    match            Instance of ``OFPMatch``
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
        def flow_removed_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.reason == ofp.OFPRR_IDLE_TIMEOUT:
                reason = 'IDLE TIMEOUT'
            elif msg.reason == ofp.OFPRR_HARD_TIMEOUT:
                reason = 'HARD TIMEOUT'
            elif msg.reason == ofp.OFPRR_DELETE:
                reason = 'DELETE'
            elif msg.reason == ofp.OFPRR_GROUP_DELETE:
                reason = 'GROUP DELETE'
            else:
                reason = 'unknown'

            self.logger.debug('OFPFlowRemoved received: '
                              'cookie=%d priority=%d reason=%s table_id=%d '
                              'duration_sec=%d duration_nsec=%d '
                              'idle_timeout=%d hard_timeout=%d '
                              'packet_count=%d byte_count=%d match.fields=%s',
                              msg.cookie, msg.priority, reason, msg.table_id,
                              msg.duration_sec, msg.duration_nsec,
                              msg.idle_timeout, msg.hard_timeout,
                              msg.packet_count, msg.byte_count, msg.match)
    """

    def __init__(self, datapath, cookie=None, priority=None, reason=None, table_id=None, duration_sec=None, duration_nsec=None, idle_timeout=None, hard_timeout=None, packet_count=None, byte_count=None, match=None):
        super(OFPFlowRemoved, self).__init__(datapath)
        self.cookie = cookie
        self.priority = priority
        self.reason = reason
        self.table_id = table_id
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.match = match

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPFlowRemoved, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.cookie, msg.priority, msg.reason, msg.table_id, msg.duration_sec, msg.duration_nsec, msg.idle_timeout, msg.hard_timeout, msg.packet_count, msg.byte_count = struct.unpack_from(ofproto.OFP_FLOW_REMOVED_PACK_STR0, msg.buf, ofproto.OFP_HEADER_SIZE)
        offset = ofproto.OFP_FLOW_REMOVED_SIZE - ofproto.OFP_MATCH_SIZE
        msg.match = OFPMatch.parser(msg.buf, offset)
        return msg