import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionCT(NXAction):
    """
        Pass traffic to the connection tracker action

        This action sends the packet through the connection tracker.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          ct(argument[,argument]...)
        ..

        +------------------------------------------------+
        | **ct(**\\ *argument*\\[,\\ *argument*\\]...\\ **)** |
        +------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        flags            Zero or more(Unspecified flag bits must be zero.)
        zone_src         OXM/NXM header for source field
        zone_ofs_nbits   Start and End for the OXM/NXM field.
                         Setting method refer to the ``nicira_ext.ofs_nbits``.
                         If you need set the Immediate value for zone,
                         zone_src must be set to None or empty character string.
        recirc_table     Recirculate to a specific table
        alg              Well-known port number for the protocol
        actions          Zero or more actions may immediately follow this
                         action
        ================ ======================================================

        .. NOTE::

            If you set number to zone_src,
            Traceback occurs when you run the to_jsondict.

        Example::

            match = parser.OFPMatch(eth_type=0x0800, ct_state=(0,32))
            actions += [parser.NXActionCT(
                            flags = 1,
                            zone_src = "reg0",
                            zone_ofs_nbits = nicira_ext.ofs_nbits(4, 31),
                            recirc_table = 4,
                            alg = 0,
                            actions = [])]
        """
    _subtype = nicira_ext.NXAST_CT
    _fmt_str = '!H4sHB3xH'
    _TYPE = {'ascii': ['zone_src']}

    def __init__(self, flags, zone_src, zone_ofs_nbits, recirc_table, alg, actions, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionCT, self).__init__()
        self.flags = flags
        self.zone_src = zone_src
        self.zone_ofs_nbits = zone_ofs_nbits
        self.recirc_table = recirc_table
        self.alg = alg
        self.actions = actions

    @classmethod
    def parser(cls, buf):
        flags, oxm_data, zone_ofs_nbits, recirc_table, alg = struct.unpack_from(cls._fmt_str, buf, 0)
        rest = buf[struct.calcsize(cls._fmt_str):]
        if oxm_data == b'\x00' * 4:
            zone_src = ''
        else:
            n, len_ = ofp.oxm_parse_header(oxm_data, 0)
            zone_src = ofp.oxm_to_user_header(n)
        actions = []
        while len(rest) > 0:
            action = ofpp.OFPAction.parser(rest, 0)
            actions.append(action)
            rest = rest[action.len:]
        return cls(flags, zone_src, zone_ofs_nbits, recirc_table, alg, actions)

    def serialize_body(self):
        data = bytearray()
        if not self.zone_src:
            zone_src = b'\x00' * 4
        elif isinstance(self.zone_src, int):
            zone_src = struct.pack('!I', self.zone_src)
        else:
            zone_src = bytearray()
            oxm = ofp.oxm_from_user_header(self.zone_src)
            ofp.oxm_serialize_header(oxm, zone_src, 0)
        msg_pack_into(self._fmt_str, data, 0, self.flags, bytes(zone_src), self.zone_ofs_nbits, self.recirc_table, self.alg)
        for a in self.actions:
            a.serialize(data, len(data))
        return data