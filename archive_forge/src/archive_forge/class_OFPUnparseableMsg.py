import struct
from os_ken.lib import stringify
from . import packet_base
class OFPUnparseableMsg(stringify.StringifyMixin):
    """Unparseable OpenFlow message encoder class.

    An instance has the following attributes at least.

    ============== ======================================================
    Attribute      Description
    ============== ======================================================
    datapath       A os_ken.ofproto.ofproto_protocol.ProtocolDesc instance
                   for this message or None if OpenFlow protocol version
                   is unsupported version.
    version        OpenFlow protocol version
    msg_type       Type of OpenFlow message
    msg_len        Length of the message
    xid            Transaction id
    body           OpenFlow body data
    ============== ======================================================

    .. Note::

        "datapath" attribute is different from
        os_ken.controller.controller.Datapath.
        So you can not use "datapath" attribute to send OpenFlow messages.
        For example, "datapath" attribute does not have send_msg method.
    """

    def __init__(self, datapath, version, msg_type, msg_len, xid, body):
        self.datapath = datapath
        self.version = version
        self.msg_type = msg_type
        self.msg_len = msg_len
        self.xid = xid
        self.body = body
        self.buf = None

    def serialize(self):
        self.buf = struct.pack(openflow.PACK_STR, self.version, self.msg_type, self.msg_len, self.xid)
        self.buf += self.body