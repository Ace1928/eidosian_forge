import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionNote(NXAction):
    """
        Note action

        This action does nothing at all.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          note:[hh]..
        ..

        +-----------------------------------+
        | **note**\\:\\ **[**\\ *hh*\\ **]**\\.. |
        +-----------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        note             A list of integer type values
        ================ ======================================================

        Example::

            actions += [parser.NXActionNote(note=[0xaa,0xbb,0xcc,0xdd])]
        """
    _subtype = nicira_ext.NXAST_NOTE
    _fmt_str = '!%dB'

    def __init__(self, note, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionNote, self).__init__()
        self.note = note

    @classmethod
    def parser(cls, buf):
        note = struct.unpack_from(cls._fmt_str % len(buf), buf, 0)
        return cls(list(note))

    def serialize_body(self):
        assert isinstance(self.note, (tuple, list))
        for n in self.note:
            assert isinstance(n, int)
        pad = (len(self.note) + nicira_ext.NX_ACTION_HEADER_0_SIZE) % 8
        if pad:
            self.note += [0 for i in range(8 - pad)]
        note_len = len(self.note)
        data = bytearray()
        msg_pack_into(self._fmt_str % note_len, data, 0, *self.note)
        return data