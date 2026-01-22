import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(MLDV2_LISTENER_REPORT)
class mldv2_report(mld):
    """
    ICMPv6 sub encoder/decoder class for MLD v2 Lister Report messages.
    (RFC 3810)

    http://www.ietf.org/rfc/rfc3810.txt

    This is used with os_ken.lib.packet.icmpv6.icmpv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== =========================================
    Attribute      Description
    ============== =========================================
    record_num     a number of the group records.
    records        a list of os_ken.lib.packet.icmpv6.mldv2_report_group.
                   None if no records.
    ============== =========================================
    """
    _PACK_STR = '!2xH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _class_prefixes = ['mldv2_report_group']

    def __init__(self, record_num=0, records=None):
        self.record_num = record_num
        records = records or []
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, mldv2_report_group)
        self.records = records

    @classmethod
    def parser(cls, buf, offset):
        record_num, = struct.unpack_from(cls._PACK_STR, buf, offset)
        offset += cls._MIN_LEN
        records = []
        while 0 < len(buf[offset:]) and record_num > len(records):
            record = mldv2_report_group.parser(buf[offset:])
            records.append(record)
            offset += len(record)
        assert record_num == len(records)
        return cls(record_num, records)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.record_num))
        for record in self.records:
            buf.extend(record.serialize())
        if 0 == self.record_num:
            self.record_num = len(self.records)
            struct.pack_into('!H', buf, 2, self.record_num)
        return bytes(buf)

    def __len__(self):
        records_len = 0
        for record in self.records:
            records_len += len(record)
        return self._MIN_LEN + records_len