from netaddr.core import NotRegisteredError, AddrFormatError, DictDotLookup
from netaddr.strategy import eui48 as _eui48, eui64 as _eui64
from netaddr.strategy.eui48 import mac_eui48
from netaddr.strategy.eui64 import eui64_base
from netaddr.ip import IPAddress
from netaddr.compat import _open_binary
class EUI(BaseIdentifier):
    """
    An IEEE EUI (Extended Unique Identifier).

    Both EUI-48 (used for layer 2 MAC addresses) and EUI-64 are supported.

    Input parsing for EUI-48 addresses is flexible, supporting many MAC
    variants.

    """
    __slots__ = ('_module', '_dialect')

    def __init__(self, addr, version=None, dialect=None):
        """
        Constructor.

        :param addr: an EUI-48 (MAC) or EUI-64 address in string format or             an unsigned integer. May also be another EUI object (copy             construction).

        :param version: (optional) the explicit EUI address version, either             48 or 64. Mainly used to distinguish EUI-48 and EUI-64 identifiers             specified as integers which may be numerically equivalent.

        :param dialect: (optional) one of the :ref:`mac_formatting_dialects` to
            be used to configure the formatting of EUI-48 (MAC) addresses.
        """
        super(EUI, self).__init__()
        self._module = None
        if isinstance(addr, EUI):
            if version is not None and version != addr._module.version:
                raise ValueError('cannot switch EUI versions using copy constructor!')
            self._module = addr._module
            self._value = addr._value
            self.dialect = dialect or addr.dialect
            return
        if version is not None:
            if version == 48:
                self._module = _eui48
            elif version == 64:
                self._module = _eui64
            else:
                raise ValueError('unsupported EUI version %r' % version)
        elif isinstance(addr, int):
            if 0 <= addr <= 281474976710655:
                self._module = _eui48
            elif 281474976710655 < addr <= 18446744073709551615:
                self._module = _eui64
        self.value = addr
        self.dialect = dialect

    def __getstate__(self):
        """:returns: Pickled state of an `EUI` object."""
        return (self._value, self._module.version, self.dialect)

    def __setstate__(self, state):
        """
        :param state: data used to unpickle a pickled `EUI` object.

        """
        value, version, dialect = state
        self._value = value
        if version == 48:
            self._module = _eui48
        elif version == 64:
            self._module = _eui64
        else:
            raise ValueError('unpickling failed for object state: %s' % (state,))
        self.dialect = dialect

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        if self._module is None:
            for module in (_eui48, _eui64):
                try:
                    self._value = module.str_to_int(value)
                    self._module = module
                    break
                except AddrFormatError:
                    try:
                        if 0 <= int(value) <= module.max_int:
                            self._value = int(value)
                            self._module = module
                            break
                    except ValueError:
                        pass
            if self._module is None:
                raise AddrFormatError('failed to detect EUI version: %r' % (value,))
        elif isinstance(value, str):
            try:
                self._value = self._module.str_to_int(value)
            except AddrFormatError:
                raise AddrFormatError('address %r is not an EUIv%d' % (value, self._module.version))
        elif 0 <= int(value) <= self._module.max_int:
            self._value = int(value)
        else:
            raise AddrFormatError('bad address format: %r' % (value,))
    value = property(_get_value, _set_value, None, 'a positive integer representing the value of this EUI identifier.')

    def _get_dialect(self):
        return self._dialect

    def _validate_dialect(self, value):
        if value is None:
            if self._module is _eui64:
                return eui64_base
            else:
                return mac_eui48
        elif hasattr(value, 'word_size') and hasattr(value, 'word_fmt'):
            return value
        else:
            raise TypeError('custom dialects should subclass mac_eui48!')

    def _set_dialect(self, value):
        self._dialect = self._validate_dialect(value)
    dialect = property(_get_dialect, _set_dialect, None, 'a Python class providing support for the interpretation of various MAC\n address formats.')

    @property
    def oui(self):
        """The OUI (Organisationally Unique Identifier) for this EUI."""
        if self._module == _eui48:
            return OUI(self.value >> 24)
        elif self._module == _eui64:
            return OUI(self.value >> 40)

    @property
    def ei(self):
        """The EI (Extension Identifier) for this EUI"""
        if self._module == _eui48:
            return '%02X-%02X-%02X' % tuple(self[3:6])
        elif self._module == _eui64:
            return '%02X-%02X-%02X-%02X-%02X' % tuple(self[3:8])

    def is_iab(self):
        """:return: True if this EUI is an IAB address, False otherwise"""
        return self._value >> 24 in IAB.IAB_EUI_VALUES

    @property
    def iab(self):
        """
        If is_iab() is True, the IAB (Individual Address Block) is returned,
        ``None`` otherwise.
        """
        if self.is_iab():
            return IAB(self._value >> 12)

    @property
    def version(self):
        """The EUI version represented by this EUI object."""
        return self._module.version

    def __getitem__(self, idx):
        """
        :return: The integer value of the word referenced by index (both             positive and negative). Raises ``IndexError`` if index is out             of bounds. Also supports Python list slices for accessing             word groups.
        """
        if isinstance(idx, int):
            num_words = self._dialect.num_words
            if not -num_words <= idx <= num_words - 1:
                raise IndexError('index out range for address type!')
            return self._module.int_to_words(self._value, self._dialect)[idx]
        elif isinstance(idx, slice):
            words = self._module.int_to_words(self._value, self._dialect)
            return [words[i] for i in range(*idx.indices(len(words)))]
        else:
            raise TypeError('unsupported type %r!' % (idx,))

    def __setitem__(self, idx, value):
        """Set the value of the word referenced by index in this address"""
        if isinstance(idx, slice):
            raise NotImplementedError('settable slices are not supported!')
        if not isinstance(idx, int):
            raise TypeError('index not an integer!')
        if not 0 <= idx <= self._dialect.num_words - 1:
            raise IndexError('index %d outside address type boundary!' % (idx,))
        if not isinstance(value, int):
            raise TypeError('value not an integer!')
        if not 0 <= value <= self._dialect.max_word:
            raise IndexError('value %d outside word size maximum of %d bits!' % (value, self._dialect.word_size))
        words = list(self._module.int_to_words(self._value, self._dialect))
        words[idx] = value
        self._value = self._module.words_to_int(words)

    def __hash__(self):
        """:return: hash of this EUI object suitable for dict keys, sets etc"""
        return hash((self.version, self._value))

    def __eq__(self, other):
        """
        :return: ``True`` if this EUI object is numerically the same as other,             ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) == (other.version, other._value)

    def __ne__(self, other):
        """
        :return: ``True`` if this EUI object is numerically the same as other,             ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) != (other.version, other._value)

    def __lt__(self, other):
        """
        :return: ``True`` if this EUI object is numerically lower in value than             other, ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) < (other.version, other._value)

    def __le__(self, other):
        """
        :return: ``True`` if this EUI object is numerically lower or equal in             value to other, ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) <= (other.version, other._value)

    def __gt__(self, other):
        """
        :return: ``True`` if this EUI object is numerically greater in value             than other, ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) > (other.version, other._value)

    def __ge__(self, other):
        """
        :return: ``True`` if this EUI object is numerically greater or equal             in value to other, ``False`` otherwise.
        """
        if not isinstance(other, EUI):
            try:
                other = self.__class__(other)
            except Exception:
                return NotImplemented
        return (self.version, self._value) >= (other.version, other._value)

    def bits(self, word_sep=None):
        """
        :param word_sep: (optional) the separator to insert between words.             Default: None - use default separator for address type.

        :return: human-readable binary digit string of this address.
        """
        return self._module.int_to_bits(self._value, word_sep)

    @property
    def packed(self):
        """The value of this EUI address as a packed binary string."""
        return self._module.int_to_packed(self._value)

    @property
    def words(self):
        """A list of unsigned integer octets found in this EUI address."""
        return self._module.int_to_words(self._value)

    @property
    def bin(self):
        """
        The value of this EUI address in standard Python binary
        representational form (0bxxx). A back port of the format provided by
        the builtin bin() function found in Python 2.6.x and higher.
        """
        return self._module.int_to_bin(self._value)

    def eui64(self):
        """
        - If this object represents an EUI-48 it is converted to EUI-64             as per the standard.
        - If this object is already an EUI-64, a new, numerically             equivalent object is returned instead.

        :return: The value of this EUI object as a new 64-bit EUI object.
        """
        if self.version == 48:
            first_three = self._value >> 24
            last_three = self._value & 16777215
            new_value = first_three << 40 | 1099478073344 | last_three
        else:
            new_value = self._value
        return self.__class__(new_value, version=64)

    def modified_eui64(self):
        """
        - create a new EUI object with a modified EUI-64 as described in RFC 4291 section 2.5.1

        :return: a new and modified 64-bit EUI object.
        """
        eui64 = self.eui64()
        eui64._value ^= 144115188075855872
        return eui64

    def ipv6(self, prefix):
        """
        .. note:: This poses security risks in certain scenarios.             Please read RFC 4941 for details. Reference: RFCs 4291 and 4941.

        :param prefix: ipv6 prefix

        :return: new IPv6 `IPAddress` object based on this `EUI`             using the technique described in RFC 4291.
        """
        int_val = int(prefix) + int(self.modified_eui64())
        return IPAddress(int_val, version=6)

    def ipv6_link_local(self):
        """
        .. note:: This poses security risks in certain scenarios.             Please read RFC 4941 for details. Reference: RFCs 4291 and 4941.

        :return: new link local IPv6 `IPAddress` object based on this `EUI`             using the technique described in RFC 4291.
        """
        return self.ipv6(338288524927261089654018896841347694592)

    @property
    def info(self):
        """
        A record dict containing IEEE registration details for this EUI
        (MAC-48) if available, None otherwise.
        """
        data = {'OUI': self.oui.registration()}
        if self.is_iab():
            data['IAB'] = self.iab.registration()
        return DictDotLookup(data)

    def format(self, dialect=None):
        """
        Format the EUI into the representational format according to the given
        dialect

        :param dialect: one of the :ref:`mac_formatting_dialects` defining the
            formatting of EUI-48 (MAC) addresses.

        :return: EUI in representational format according to the given dialect
        """
        validated_dialect = self._validate_dialect(dialect)
        return self._module.int_to_str(self._value, validated_dialect)

    def __str__(self):
        """:return: EUI in representational format"""
        return self._module.int_to_str(self._value, self._dialect)

    def __repr__(self):
        """:return: executable Python string to recreate equivalent object."""
        return "EUI('%s')" % self