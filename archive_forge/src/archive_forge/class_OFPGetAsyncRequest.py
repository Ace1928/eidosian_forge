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
@_set_msg_type(ofproto.OFPT_GET_ASYNC_REQUEST)
class OFPGetAsyncRequest(MsgBase):
    """
    Get asynchronous configuration request message

    The controller uses this message to query the asynchronous message.

    Example::

        def send_get_async_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPGetAsyncRequest(datapath)
            datapath.send_msg(req)
    """

    def __init__(self, datapath):
        super(OFPGetAsyncRequest, self).__init__(datapath)