import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
class tcp(packet_base.PacketBase):
    """TCP (RFC 793) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    src_port       Source Port
    dst_port       Destination Port
    seq            Sequence Number
    ack            Acknowledgement Number
    offset         Data Offset                    (0 means automatically-calculate when encoding)
    bits           Control Bits
    window_size    Window
    csum           Checksum                    (0 means automatically-calculate when encoding)
    urgent         Urgent Pointer
    option         List of ``TCPOption`` sub-classes or an bytearray
                   containing options.                    None if no options.
    ============== ====================
    """
    _PACK_STR = '!HHIIBBHHH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, src_port=1, dst_port=1, seq=0, ack=0, offset=0, bits=0, window_size=0, csum=0, urgent=0, option=None):
        super(tcp, self).__init__()
        self.src_port = src_port
        self.dst_port = dst_port
        self.seq = seq
        self.ack = ack
        self.offset = offset
        self.bits = bits
        self.window_size = window_size
        self.csum = csum
        self.urgent = urgent
        self.option = option

    def __len__(self):
        return self.offset * 4

    def has_flags(self, *flags):
        """Check if flags are set on this packet.

        returns boolean if all passed flags is set

        Example::

            >>> pkt = tcp.tcp(bits=(tcp.TCP_SYN | tcp.TCP_ACK))
            >>> pkt.has_flags(tcp.TCP_SYN, tcp.TCP_ACK)
            True
        """
        mask = sum(flags)
        return self.bits & mask == mask

    @staticmethod
    def get_payload_type(src_port, dst_port):
        from os_ken.ofproto.ofproto_common import OFP_TCP_PORT, OFP_SSL_PORT_OLD
        if bgp.TCP_SERVER_PORT in [src_port, dst_port]:
            return bgp.BGPMessage
        elif src_port in [OFP_TCP_PORT, OFP_SSL_PORT_OLD] or dst_port in [OFP_TCP_PORT, OFP_SSL_PORT_OLD]:
            return openflow.openflow
        elif src_port == zebra.ZEBRA_PORT:
            return zebra._ZebraMessageFromZebra
        elif dst_port == zebra.ZEBRA_PORT:
            return zebra.ZebraMessage
        else:
            return None

    @classmethod
    def parser(cls, buf):
        src_port, dst_port, seq, ack, offset, bits, window_size, csum, urgent = struct.unpack_from(cls._PACK_STR, buf)
        offset >>= 4
        bits &= 63
        length = offset * 4
        if length > tcp._MIN_LEN:
            option_buf = buf[tcp._MIN_LEN:length]
            try:
                option = []
                while option_buf:
                    opt, option_buf = TCPOption.parser(option_buf)
                    option.append(opt)
            except struct.error:
                LOG.warning('Encounter an error during parsing TCP option field.Skip parsing TCP option.')
                option = buf[tcp._MIN_LEN:length]
        else:
            option = None
        msg = cls(src_port, dst_port, seq, ack, offset, bits, window_size, csum, urgent, option)
        return (msg, cls.get_payload_type(src_port, dst_port), buf[length:])

    def serialize(self, payload, prev):
        offset = self.offset << 4
        h = bytearray(struct.pack(tcp._PACK_STR, self.src_port, self.dst_port, self.seq, self.ack, offset, self.bits, self.window_size, self.csum, self.urgent))
        if self.option:
            if isinstance(self.option, (list, tuple)):
                option_buf = bytearray()
                for opt in self.option:
                    option_buf.extend(opt.serialize())
                h.extend(option_buf)
                mod = len(option_buf) % 4
            else:
                h.extend(self.option)
                mod = len(self.option) % 4
            if mod:
                h.extend(bytearray(4 - mod))
            if self.offset:
                offset = self.offset << 2
                if len(h) < offset:
                    h.extend(bytearray(offset - len(h)))
        if self.offset == 0:
            self.offset = len(h) >> 2
            offset = self.offset << 4
            struct.pack_into('!B', h, 12, offset)
        if self.csum == 0:
            total_length = len(h) + len(payload)
            self.csum = packet_utils.checksum_ip(prev, total_length, h + payload)
            struct.pack_into('!H', h, 16, self.csum)
        return bytes(h)