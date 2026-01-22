import random
import struct
from . import packet_base
from os_ken.lib import addrconv
from os_ken.lib import stringify
class dhcp6(packet_base.PacketBase):
    """DHCPv6 (RFC 3315) header encoder/decoder class.

    The serialized packet would looks like the ones described
    in the following sections.

    * RFC 3315 DHCP packet format

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.


    ============== ====================
    Attribute      Description
    ============== ====================
    msg_type       Identifies the DHCP message type
    transaction_id For unrelayed messages only: the transaction ID for                   this message exchange.
    hop_count      For relayed messages only: number of relay agents that                   have relayed this message.
    link_address   For relayed messages only: a global or site-local address                   that will be used by the server to identify the link on                   which the client is located.
    peer_address   For relayed messages only: the address of the client or                   relay agent from which the message to be relayed was                   received.
    options        Options carried in this message
    ============== ====================
    """
    _MIN_LEN = 8
    _DHCPV6_UNPACK_STR = '!I'
    _DHCPV6_RELAY_UNPACK_STR = '!H16s16s'
    _DHCPV6_UNPACK_STR_LEN = struct.calcsize(_DHCPV6_UNPACK_STR)
    _DHCPV6_RELAY_UNPACK_STR_LEN = struct.calcsize(_DHCPV6_RELAY_UNPACK_STR)
    _DHCPV6_PACK_STR = '!I'
    _DHCPV6_RELAY_PACK_STR = '!H16s16s'

    def __init__(self, msg_type, options, transaction_id=None, hop_count=0, link_address='::', peer_address='::'):
        super(dhcp6, self).__init__()
        self.msg_type = msg_type
        self.options = options
        if transaction_id is None:
            self.transaction_id = random.randint(0, 16777215)
        else:
            self.transaction_id = transaction_id
        self.hop_count = hop_count
        self.link_address = link_address
        self.peer_address = peer_address

    @classmethod
    def parser(cls, buf):
        msg_type, = struct.unpack_from('!B', buf)
        buf = b'\x00' + buf[1:]
        if msg_type == DHCPV6_RELAY_FORW or msg_type == DHCPV6_RELAY_REPL:
            hop_count, link_address, peer_address = struct.unpack_from(cls._DHCPV6_RELAY_UNPACK_STR, buf)
            length = struct.calcsize(cls._DHCPV6_RELAY_UNPACK_STR)
        else:
            transaction_id, = struct.unpack_from(cls._DHCPV6_UNPACK_STR, buf)
            length = struct.calcsize(cls._DHCPV6_UNPACK_STR)
        if len(buf) > length:
            parse_opt = options.parser(buf[length:])
            length += parse_opt.options_len
            if msg_type == DHCPV6_RELAY_FORW or msg_type == DHCPV6_RELAY_REPL:
                return (cls(msg_type, parse_opt, 0, hop_count, addrconv.ipv6.bin_to_text(link_address), addrconv.ipv6.bin_to_text(peer_address)), None, buf[length:])
            else:
                return (cls(msg_type, parse_opt, transaction_id), None, buf[length:])
        else:
            return (None, None, buf)

    def serialize(self, payload=None, prev=None):
        seri_opt = self.options.serialize()
        if self.msg_type == DHCPV6_RELAY_FORW or self.msg_type == DHCPV6_RELAY_REPL:
            pack_str = '%s%ds' % (self._DHCPV6_RELAY_PACK_STR, self.options.options_len)
            buf = struct.pack(pack_str, self.hop_count, addrconv.ipv6.text_to_bin(self.link_address), addrconv.ipv6.text_to_bin(self.peer_address), seri_opt)
        else:
            pack_str = '%s%ds' % (self._DHCPV6_PACK_STR, self.options.options_len)
            buf = struct.pack(pack_str, self.transaction_id, seri_opt)
        return struct.pack('!B', self.msg_type) + buf[1:]