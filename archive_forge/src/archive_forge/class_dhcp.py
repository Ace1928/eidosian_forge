import random
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import stringify
from . import packet_base
class dhcp(packet_base.PacketBase):
    """DHCP (RFC 2131) header encoder/decoder class.

    The serialized packet would looks like the ones described
    in the following sections.

    * RFC 2131 DHCP packet format

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== ====================
    Attribute      Description
    ============== ====================
    op             Message op code / message type.                   1 = BOOTREQUEST, 2 = BOOTREPLY
    htype          Hardware address type (e.g.  '1' = 10mb ethernet).
    hlen           Hardware address length (e.g.  '6' = 10mb ethernet).
    hops           Client sets to zero, optionally used by relay agent                   when booting via a relay agent.
    xid            Transaction ID, a random number chosen by the client,                   used by the client and serverto associate messages                   and responses between a client and a server.
    secs           Filled in by client, seconds elapsed since client                   began address acquisition or renewal process.
    flags          Flags.
    ciaddr         Client IP address; only filled in if client is in                   BOUND, RENEW or REBINDING state and can respond                   to ARP requests.
    yiaddr         'your' (client) IP address.
    siaddr         IP address of next server to use in bootstrap;                   returned in DHCPOFFER, DHCPACK by server.
    giaddr         Relay agent IP address, used in booting via a                   relay agent.
    chaddr         Client hardware address.
    sname          Optional server host name, null terminated string.
    boot_file      Boot file name, null terminated string; "generic"                   name or null in DHCPDISCOVER, fully qualified                   directory-path name in DHCPOFFER.
    options        Optional parameters field                   ('DHCP message type' option must be included in                    every DHCP message).
    ============== ====================
    """
    _DHCP_PACK_STR = '!BBBBIHH4s4s4s4s16s64s128s'
    _MIN_LEN = struct.calcsize(_DHCP_PACK_STR)
    _MAC_ADDRESS_LEN = 6
    _HARDWARE_TYPE_ETHERNET = 1
    _class_prefixes = ['options']
    _TYPE = {'ascii': ['ciaddr', 'yiaddr', 'siaddr', 'giaddr', 'chaddr', 'sname', 'boot_file']}

    def __init__(self, op, chaddr, options=None, htype=_HARDWARE_TYPE_ETHERNET, hlen=0, hops=0, xid=None, secs=0, flags=0, ciaddr='0.0.0.0', yiaddr='0.0.0.0', siaddr='0.0.0.0', giaddr='0.0.0.0', sname='', boot_file=''):
        super(dhcp, self).__init__()
        self.op = op
        self.htype = htype
        self.hlen = hlen
        self.hops = hops
        if xid is None:
            self.xid = random.randint(0, 4294967295)
        else:
            self.xid = xid
        self.secs = secs
        self.flags = flags
        self.ciaddr = ciaddr
        self.yiaddr = yiaddr
        self.siaddr = siaddr
        self.giaddr = giaddr
        self.chaddr = chaddr
        self.sname = sname
        self.boot_file = boot_file
        self.options = options

    @classmethod
    def parser(cls, buf):
        op, htype, hlen, hops, xid, secs, flags, ciaddr, yiaddr, siaddr, giaddr, chaddr, sname, boot_file = struct.unpack_from(cls._DHCP_PACK_STR, buf)
        if hlen == cls._MAC_ADDRESS_LEN:
            chaddr = addrconv.mac.bin_to_text(chaddr[:cls._MAC_ADDRESS_LEN])
        length = cls._MIN_LEN
        parse_opt = None
        if len(buf) > length:
            parse_opt = options.parser(buf[length:])
            length += parse_opt.options_len
        return (cls(op, chaddr, parse_opt, htype, hlen, hops, xid, secs, flags, addrconv.ipv4.bin_to_text(ciaddr), addrconv.ipv4.bin_to_text(yiaddr), addrconv.ipv4.bin_to_text(siaddr), addrconv.ipv4.bin_to_text(giaddr), sname.decode('ascii'), boot_file.decode('ascii')), None, buf[length:])

    def serialize(self, _payload=None, _prev=None):
        opt_buf = bytearray()
        if self.options is not None:
            opt_buf = self.options.serialize()
        if netaddr.valid_mac(self.chaddr):
            chaddr = addrconv.mac.text_to_bin(self.chaddr)
        else:
            chaddr = self.chaddr
        self.hlen = len(chaddr)
        return struct.pack(self._DHCP_PACK_STR, self.op, self.htype, self.hlen, self.hops, self.xid, self.secs, self.flags, addrconv.ipv4.text_to_bin(self.ciaddr), addrconv.ipv4.text_to_bin(self.yiaddr), addrconv.ipv4.text_to_bin(self.siaddr), addrconv.ipv4.text_to_bin(self.giaddr), chaddr, self.sname.encode('ascii'), self.boot_file.encode('ascii')) + opt_buf