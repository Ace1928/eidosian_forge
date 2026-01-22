import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionNAT(NXAction):
    """
        Network address translation action

        This action sends the packet through the connection tracker.

        And equivalent to the followings action of ovs-ofctl command.

        .. NOTE::
            The following command image does not exist in ovs-ofctl command
            manual and has been created from the command response.

        ..
          nat(src=ip_min-ip_max : proto_min-proto-max)
        ..

        +--------------------------------------------------+
        | **nat(src**\\=\\ *ip_min*\\ **-**\\ *ip_max*\\  **:** |
        | *proto_min*\\ **-**\\ *proto-max*\\ **)**           |
        +--------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        flags            Zero or more(Unspecified flag bits must be zero.)
        range_ipv4_min   Range ipv4 address minimun
        range_ipv4_max   Range ipv4 address maximun
        range_ipv6_min   Range ipv6 address minimun
        range_ipv6_max   Range ipv6 address maximun
        range_proto_min  Range protocol minimum
        range_proto_max  Range protocol maximun
        ================ ======================================================

        .. CAUTION::
            ``NXActionNAT`` must be defined in the actions in the
            ``NXActionCT``.

        Example::

            match = parser.OFPMatch(eth_type=0x0800)
            actions += [
                parser.NXActionCT(
                    flags = 1,
                    zone_src = "reg0",
                    zone_ofs_nbits = nicira_ext.ofs_nbits(4, 31),
                    recirc_table = 255,
                    alg = 0,
                    actions = [
                        parser.NXActionNAT(
                            flags = 1,
                            range_ipv4_min = "10.1.12.0",
                            range_ipv4_max = "10.1.13.255",
                            range_ipv6_min = "",
                            range_ipv6_max = "",
                            range_proto_min = 1,
                            range_proto_max = 1023
                        )
                    ]
                )
            ]
        """
    _subtype = nicira_ext.NXAST_NAT
    _fmt_str = '!2xHH'
    _TYPE = {'ascii': ['range_ipv4_max', 'range_ipv4_min', 'range_ipv6_max', 'range_ipv6_min']}

    def __init__(self, flags, range_ipv4_min='', range_ipv4_max='', range_ipv6_min='', range_ipv6_max='', range_proto_min=None, range_proto_max=None, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionNAT, self).__init__()
        self.flags = flags
        self.range_ipv4_min = range_ipv4_min
        self.range_ipv4_max = range_ipv4_max
        self.range_ipv6_min = range_ipv6_min
        self.range_ipv6_max = range_ipv6_max
        self.range_proto_min = range_proto_min
        self.range_proto_max = range_proto_max

    @classmethod
    def parser(cls, buf):
        flags, range_present = struct.unpack_from(cls._fmt_str, buf, 0)
        rest = buf[struct.calcsize(cls._fmt_str):]
        kwargs = dict()
        if range_present & nicira_ext.NX_NAT_RANGE_IPV4_MIN:
            kwargs['range_ipv4_min'] = type_desc.IPv4Addr.to_user(rest[:4])
            rest = rest[4:]
        if range_present & nicira_ext.NX_NAT_RANGE_IPV4_MAX:
            kwargs['range_ipv4_max'] = type_desc.IPv4Addr.to_user(rest[:4])
            rest = rest[4:]
        if range_present & nicira_ext.NX_NAT_RANGE_IPV6_MIN:
            kwargs['range_ipv6_min'] = type_desc.IPv6Addr.to_user(rest[:16])
            rest = rest[16:]
        if range_present & nicira_ext.NX_NAT_RANGE_IPV6_MAX:
            kwargs['range_ipv6_max'] = type_desc.IPv6Addr.to_user(rest[:16])
            rest = rest[16:]
        if range_present & nicira_ext.NX_NAT_RANGE_PROTO_MIN:
            kwargs['range_proto_min'] = type_desc.Int2.to_user(rest[:2])
            rest = rest[2:]
        if range_present & nicira_ext.NX_NAT_RANGE_PROTO_MAX:
            kwargs['range_proto_max'] = type_desc.Int2.to_user(rest[:2])
        return cls(flags, **kwargs)

    def serialize_body(self):
        optional_data = b''
        range_present = 0
        if self.range_ipv4_min != '':
            range_present |= nicira_ext.NX_NAT_RANGE_IPV4_MIN
            optional_data += type_desc.IPv4Addr.from_user(self.range_ipv4_min)
        if self.range_ipv4_max != '':
            range_present |= nicira_ext.NX_NAT_RANGE_IPV4_MAX
            optional_data += type_desc.IPv4Addr.from_user(self.range_ipv4_max)
        if self.range_ipv6_min != '':
            range_present |= nicira_ext.NX_NAT_RANGE_IPV6_MIN
            optional_data += type_desc.IPv6Addr.from_user(self.range_ipv6_min)
        if self.range_ipv6_max != '':
            range_present |= nicira_ext.NX_NAT_RANGE_IPV6_MAX
            optional_data += type_desc.IPv6Addr.from_user(self.range_ipv6_max)
        if self.range_proto_min is not None:
            range_present |= nicira_ext.NX_NAT_RANGE_PROTO_MIN
            optional_data += type_desc.Int2.from_user(self.range_proto_min)
        if self.range_proto_max is not None:
            range_present |= nicira_ext.NX_NAT_RANGE_PROTO_MAX
            optional_data += type_desc.Int2.from_user(self.range_proto_max)
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.flags, range_present)
        msg_pack_into('!%ds' % len(optional_data), data, len(data), optional_data)
        return data