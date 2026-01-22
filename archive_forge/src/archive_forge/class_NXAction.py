import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXAction(ofpp.OFPActionExperimenter):
    _fmt_str = '!H'
    _subtypes = {}
    _experimenter = ofproto_common.NX_EXPERIMENTER_ID

    def __init__(self):
        super(NXAction, self).__init__(self._experimenter)
        self.subtype = self._subtype

    @classmethod
    def parse(cls, buf):
        fmt_str = NXAction._fmt_str
        subtype, = struct.unpack_from(fmt_str, buf, 0)
        subtype_cls = cls._subtypes.get(subtype)
        rest = buf[struct.calcsize(fmt_str):]
        if subtype_cls is None:
            return NXActionUnknown(subtype, rest)
        return subtype_cls.parser(rest)

    def serialize(self, buf, offset):
        data = self.serialize_body()
        payload_offset = ofp.OFP_ACTION_EXPERIMENTER_HEADER_SIZE + struct.calcsize(NXAction._fmt_str)
        self.len = utils.round_up(payload_offset + len(data), 8)
        super(NXAction, self).serialize(buf, offset)
        msg_pack_into(NXAction._fmt_str, buf, offset + ofp.OFP_ACTION_EXPERIMENTER_HEADER_SIZE, self.subtype)
        buf += data

    @classmethod
    def register(cls, subtype_cls):
        assert subtype_cls._subtype is not cls._subtypes
        cls._subtypes[subtype_cls._subtype] = subtype_cls