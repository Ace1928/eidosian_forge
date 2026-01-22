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
@_set_msg_type(ofproto.OFPT_PORT_STATUS)
class OFPPortStatus(MsgBase):
    """
    Port status message

    The switch notifies controller of change of ports.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    reason           One of the following values.

                     | OFPPR_ADD
                     | OFPPR_DELETE
                     | OFPPR_MODIFY
    desc             instance of ``OFPPort``
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
        def port_status_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.reason == ofp.OFPPR_ADD:
                reason = 'ADD'
            elif msg.reason == ofp.OFPPR_DELETE:
                reason = 'DELETE'
            elif msg.reason == ofp.OFPPR_MODIFY:
                reason = 'MODIFY'
            else:
                reason = 'unknown'

            self.logger.debug('OFPPortStatus received: reason=%s desc=%s',
                              reason, msg.desc)
    """

    def __init__(self, datapath, reason=None, desc=None):
        super(OFPPortStatus, self).__init__(datapath)
        self.reason = reason
        self.desc = desc

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPPortStatus, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.reason = struct.unpack_from(ofproto.OFP_PORT_STATUS_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)[0]
        msg.desc = OFPPort.parser(msg.buf, ofproto.OFP_PORT_STATUS_DESC_OFFSET)
        return msg