import functools
@functools.total_ordering
class _BaseAddress(_IPAddressBase):
    """A generic IP object.

    This IP class contains the version independent methods which are
    used by single IP addresses.
    """
    __slots__ = ()

    def __int__(self):
        return self._ip

    def __eq__(self, other):
        try:
            return self._ip == other._ip and self._version == other._version
        except AttributeError:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, _BaseAddress):
            return NotImplemented
        if self._version != other._version:
            raise TypeError('%s and %s are not of the same version' % (self, other))
        if self._ip != other._ip:
            return self._ip < other._ip
        return False

    def __add__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return self.__class__(int(self) + other)

    def __sub__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return self.__class__(int(self) - other)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __str__(self):
        return str(self._string_from_ip_int(self._ip))

    def __hash__(self):
        return hash(hex(int(self._ip)))

    def _get_address_key(self):
        return (self._version, self)

    def __reduce__(self):
        return (self.__class__, (self._ip,))

    def __format__(self, fmt):
        """Returns an IP address as a formatted string.

        Supported presentation types are:
        's': returns the IP address as a string (default)
        'b': converts to binary and returns a zero-padded string
        'X' or 'x': converts to upper- or lower-case hex and returns a zero-padded string
        'n': the same as 'b' for IPv4 and 'x' for IPv6

        For binary and hex presentation types, the alternate form specifier
        '#' and the grouping option '_' are supported.
        """
        if not fmt or fmt[-1] == 's':
            return format(str(self), fmt)
        global _address_fmt_re
        if _address_fmt_re is None:
            import re
            _address_fmt_re = re.compile('(#?)(_?)([xbnX])')
        m = _address_fmt_re.fullmatch(fmt)
        if not m:
            return super().__format__(fmt)
        alternate, grouping, fmt_base = m.groups()
        if fmt_base == 'n':
            if self._version == 4:
                fmt_base = 'b'
            else:
                fmt_base = 'x'
        if fmt_base == 'b':
            padlen = self._max_prefixlen
        else:
            padlen = self._max_prefixlen // 4
        if grouping:
            padlen += padlen // 4 - 1
        if alternate:
            padlen += 2
        return format(int(self), f'{alternate}0{padlen}{grouping}{fmt_base}')