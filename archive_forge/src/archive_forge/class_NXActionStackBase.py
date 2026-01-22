import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionStackBase(NXAction):
    _fmt_str = '!H4sH'
    _TYPE = {'ascii': ['field']}

    def __init__(self, field, start, end, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionStackBase, self).__init__()
        self.field = field
        self.start = start
        self.end = end

    @classmethod
    def parser(cls, buf):
        start, oxm_data, end = struct.unpack_from(cls._fmt_str, buf, 0)
        n, len_ = ofp.oxm_parse_header(oxm_data, 0)
        field = ofp.oxm_to_user_header(n)
        return cls(field, start, end)

    def serialize_body(self):
        data = bytearray()
        oxm_data = bytearray()
        oxm = ofp.oxm_from_user_header(self.field)
        ofp.oxm_serialize_header(oxm, oxm_data, 0)
        msg_pack_into(self._fmt_str, data, 0, self.start, bytes(oxm_data), self.end)
        offset = len(data)
        msg_pack_into('!%dx' % (12 - offset), data, offset)
        return data