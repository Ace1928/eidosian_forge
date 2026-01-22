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
@_set_msg_type(ofproto.OFPT_FEATURES_REPLY)
class OFPSwitchFeatures(MsgBase):
    """
    Features reply message

    The switch responds with a features reply message to a features
    request.

    This message is handled by the OSKen framework, so the OSKen application
    do not need to process this typically.

    Example::

        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev):
            msg = ev.msg

            self.logger.debug('OFPSwitchFeatures received: '
                              'datapath_id=0x%016x n_buffers=%d '
                              'n_tables=%d auxiliary_id=%d '
                              'capabilities=0x%08x',
                              msg.datapath_id, msg.n_buffers, msg.n_tables,
                              msg.auxiliary_id, msg.capabilities)
    """

    def __init__(self, datapath, datapath_id=None, n_buffers=None, n_tables=None, auxiliary_id=None, capabilities=None):
        super(OFPSwitchFeatures, self).__init__(datapath)
        self.datapath_id = datapath_id
        self.n_buffers = n_buffers
        self.n_tables = n_tables
        self.auxiliary_id = auxiliary_id
        self.capabilities = capabilities

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPSwitchFeatures, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.datapath_id, msg.n_buffers, msg.n_tables, msg.auxiliary_id, msg.capabilities, msg._reserved = struct.unpack_from(ofproto.OFP_SWITCH_FEATURES_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        return msg