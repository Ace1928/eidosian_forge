import struct
from os_ken.lib import stringify
from . import packet_base
class openflow(packet_base.PacketBase):
    """OpenFlow message encoder/decoder class.

    An instance has the following attributes at least.

    ============== =========================================================
    Attribute      Description
    ============== =========================================================
    msg            An instance of OpenFlow message (see :ref:`ofproto_ref`)
                   or an instance of OFPUnparseableMsg if failed to parse
                   packet as OpenFlow message.
    ============== =========================================================
    """
    PACK_STR = '!BBHI'
    _MIN_LEN = struct.calcsize(PACK_STR)

    def __init__(self, msg):
        super(openflow, self).__init__()
        self.msg = msg

    @classmethod
    def parser(cls, buf):
        from os_ken.ofproto import ofproto_parser
        from os_ken.ofproto import ofproto_protocol
        version, msg_type, msg_len, xid = ofproto_parser.header(buf)
        msg_parser = ofproto_parser._MSG_PARSERS.get(version)
        if msg_parser is None:
            msg = OFPUnparseableMsg(None, version, msg_type, msg_len, xid, buf[cls._MIN_LEN:msg_len])
            return (cls(msg), cls, buf[msg_len:])
        datapath = ofproto_protocol.ProtocolDesc(version=version)
        try:
            msg = msg_parser(datapath, version, msg_type, msg_len, xid, buf[:msg_len])
        except:
            msg = OFPUnparseableMsg(datapath, version, msg_type, msg_len, xid, buf[datapath.ofproto.OFP_HEADER_SIZE:msg_len])
        return (cls(msg), cls, buf[msg_len:])

    def serialize(self, _payload, _prev):
        self.msg.serialize()
        return self.msg.buf