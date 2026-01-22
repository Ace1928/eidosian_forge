import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class _NXActionBundleBase(NXAction):
    _fmt_str = '!HHHIHH'

    def __init__(self, algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst, slaves):
        super(_NXActionBundleBase, self).__init__()
        self.len = utils.round_up(nicira_ext.NX_ACTION_BUNDLE_0_SIZE + len(slaves) * 2, 8)
        self.algorithm = algorithm
        self.fields = fields
        self.basis = basis
        self.slave_type = slave_type
        self.n_slaves = n_slaves
        self.ofs_nbits = ofs_nbits
        self.dst = dst
        assert isinstance(slaves, (list, tuple))
        for s in slaves:
            assert isinstance(s, int)
        self.slaves = slaves

    @classmethod
    def parser(cls, buf):
        algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst = struct.unpack_from(cls._fmt_str + 'I', buf, 0)
        offset = nicira_ext.NX_ACTION_BUNDLE_0_SIZE - nicira_ext.NX_ACTION_HEADER_0_SIZE - 8
        if dst != 0:
            n, len_ = ofp.oxm_parse_header(buf, offset)
            dst = ofp.oxm_to_user_header(n)
        slave_offset = nicira_ext.NX_ACTION_BUNDLE_0_SIZE - nicira_ext.NX_ACTION_HEADER_0_SIZE
        slaves = []
        for i in range(0, n_slaves):
            s = struct.unpack_from('!H', buf, slave_offset)
            slaves.append(s[0])
            slave_offset += 2
        return cls(algorithm, fields, basis, slave_type, n_slaves, ofs_nbits, dst, slaves)

    def serialize_body(self):
        data = bytearray()
        slave_offset = nicira_ext.NX_ACTION_BUNDLE_0_SIZE - nicira_ext.NX_ACTION_HEADER_0_SIZE
        self.n_slaves = len(self.slaves)
        for s in self.slaves:
            msg_pack_into('!H', data, slave_offset, s)
            slave_offset += 2
        pad_len = utils.round_up(self.n_slaves, 4) - self.n_slaves
        if pad_len != 0:
            msg_pack_into('%dx' % pad_len * 2, data, slave_offset)
        msg_pack_into(self._fmt_str, data, 0, self.algorithm, self.fields, self.basis, self.slave_type, self.n_slaves, self.ofs_nbits)
        offset = nicira_ext.NX_ACTION_BUNDLE_0_SIZE - nicira_ext.NX_ACTION_HEADER_0_SIZE - 8
        if self.dst == 0:
            msg_pack_into('I', data, offset, self.dst)
        else:
            oxm_data = ofp.oxm_from_user_header(self.dst)
            ofp.oxm_serialize_header(oxm_data, data, offset)
        return data