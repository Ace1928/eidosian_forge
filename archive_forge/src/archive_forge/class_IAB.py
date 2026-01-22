from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
class IAB(BaseIdentifier):
    IAB_EUI_VALUES = (20674, 4249685)
    '\n    An individual IEEE IAB (Individual Address Block) identifier.\n\n    For online details see - http://standards.ieee.org/regauth/oui/\n\n    '
    __slots__ = ('record',)

    @classmethod
    def split_iab_mac(cls, eui_int, strict=False):
        """
        :param eui_int: a MAC IAB as an unsigned integer.

        :param strict: If True, raises a ValueError if the last 12 bits of
            IAB MAC/EUI-48 address are non-zero, ignores them otherwise.
            (Default: False)
        """
        if eui_int >> 12 in cls.IAB_EUI_VALUES:
            return (eui_int, 0)
        user_mask = 2 ** 12 - 1
        iab_mask = 2 ** 48 - 1 ^ user_mask
        iab_bits = eui_int >> 12
        user_bits = (eui_int | iab_mask) - iab_mask
        if iab_bits >> 12 in cls.IAB_EUI_VALUES:
            if strict and user_bits != 0:
                raise ValueError('%r is not a strict IAB!' % hex(user_bits))
        else:
            raise ValueError('%r is not an IAB address!' % hex(eui_int))
        return (iab_bits, user_bits)

    def __init__(self, iab, strict=False):
        """
        Constructor

        :param iab: an IAB string ``00-50-C2-XX-X0-00`` or an unsigned             integer. This address looks like an EUI-48 but it should not             have any non-zero bits in the last 3 bytes.

        :param strict: If True, raises a ValueError if the last 12 bits             of IAB MAC/EUI-48 address are non-zero, ignores them otherwise.             (Default: False)
        """
        super(IAB, self).__init__()
        from netaddr.eui import ieee
        self.record = {'idx': 0, 'iab': '', 'org': '', 'address': [], 'offset': 0, 'size': 0}
        if isinstance(iab, str):
            int_val = int(iab.replace('-', ''), 16)
            iab_int, user_int = self.split_iab_mac(int_val, strict=strict)
            self._value = iab_int
        elif isinstance(iab, int):
            iab_int, user_int = self.split_iab_mac(iab, strict=strict)
            self._value = iab_int
        else:
            raise TypeError('unexpected IAB format: %r!' % (iab,))
        if self._value in ieee.IAB_INDEX:
            fh = _open_binary(__package__, 'iab.txt')
            offset, size = ieee.IAB_INDEX[self._value][0]
            self.record['offset'] = offset
            self.record['size'] = size
            fh.seek(offset)
            data = fh.read(size).decode('UTF-8')
            self._parse_data(data, offset, size)
            fh.close()
        else:
            raise NotRegisteredError('IAB %r not unregistered!' % (iab,))

    def __eq__(self, other):
        if not isinstance(other, IAB):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return self._value == other._value

    def __ne__(self, other):
        if not isinstance(other, IAB):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return self._value != other._value

    def __getstate__(self):
        """:returns: Pickled state of an `IAB` object."""
        return (self._value, self.record)

    def __setstate__(self, state):
        """:param state: data used to unpickle a pickled `IAB` object."""
        self._value, self.record = state

    def _parse_data(self, data, offset, size):
        """Returns a dict record from raw IAB record data"""
        for line in data.split('\n'):
            line = line.strip()
            if not line:
                continue
            if '(hex)' in line:
                self.record['idx'] = self._value
                self.record['org'] = line.split(None, 2)[2]
                self.record['iab'] = str(self)
            elif '(base 16)' in line:
                continue
            else:
                self.record['address'].append(line)

    def registration(self):
        """The IEEE registration details for this IAB"""
        return DictDotLookup(self.record)

    def __str__(self):
        """:return: string representation of this IAB"""
        int_val = self._value << 4
        return '%02X-%02X-%02X-%02X-%02X-00' % (int_val >> 32 & 255, int_val >> 24 & 255, int_val >> 16 & 255, int_val >> 8 & 255, int_val & 255)

    def __repr__(self):
        """:return: executable Python string to recreate equivalent object."""
        return "IAB('%s')" % self