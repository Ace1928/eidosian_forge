import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@_register_parser
@_set_msg_type(ofproto.OFPT_TABLE_STATUS)
class OFPTableStatus(MsgBase):
    """
    Table status message

    The switch notifies controller of change of table status.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    reason           One of the following values.

                     | OFPTR_VACANCY_DOWN
                     | OFPTR_VACANCY_UP
    table            ``OFPTableDesc`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPTableStatus, MAIN_DISPATCHER)
        def table(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.reason == ofp.OFPTR_VACANCY_DOWN:
                reason = 'VACANCY_DOWN'
            elif msg.reason == ofp.OFPTR_VACANCY_UP:
                reason = 'VACANCY_UP'
            else:
                reason = 'unknown'

            self.logger.debug('OFPTableStatus received: reason=%s '
                              'table_id=%d config=0x%08x properties=%s',
                              reason, msg.table.table_id, msg.table.config,
                              repr(msg.table.properties))
    """

    def __init__(self, datapath, reason=None, table=None):
        super(OFPTableStatus, self).__init__(datapath)
        self.reason = reason
        self.table = table

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPTableStatus, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.reason, = struct.unpack_from(ofproto.OFP_TABLE_STATUS_0_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        msg.table = OFPTableDesc.parser(msg.buf, ofproto.OFP_TABLE_STATUS_0_SIZE)
        return msg