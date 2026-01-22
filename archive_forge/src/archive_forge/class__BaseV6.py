import functools
class _BaseV6:
    """Base IPv6 object.

    The following methods are used by IPv6 objects in both single IP
    addresses and networks.

    """
    __slots__ = ()
    _version = 6
    _ALL_ONES = 2 ** IPV6LENGTH - 1
    _HEXTET_COUNT = 8
    _HEX_DIGITS = frozenset('0123456789ABCDEFabcdef')
    _max_prefixlen = IPV6LENGTH
    _netmask_cache = {}

    @classmethod
    def _make_netmask(cls, arg):
        """Make a (netmask, prefix_len) tuple from the given argument.

        Argument can be:
        - an integer (the prefix length)
        - a string representing the prefix length (e.g. "24")
        - a string representing the prefix netmask (e.g. "255.255.255.0")
        """
        if arg not in cls._netmask_cache:
            if isinstance(arg, int):
                prefixlen = arg
                if not 0 <= prefixlen <= cls._max_prefixlen:
                    cls._report_invalid_netmask(prefixlen)
            else:
                prefixlen = cls._prefix_from_prefix_string(arg)
            netmask = IPv6Address(cls._ip_int_from_prefix(prefixlen))
            cls._netmask_cache[arg] = (netmask, prefixlen)
        return cls._netmask_cache[arg]

    @classmethod
    def _ip_int_from_string(cls, ip_str):
        """Turn an IPv6 ip_str into an integer.

        Args:
            ip_str: A string, the IPv6 ip_str.

        Returns:
            An int, the IPv6 address

        Raises:
            AddressValueError: if ip_str isn't a valid IPv6 Address.

        """
        if not ip_str:
            raise AddressValueError('Address cannot be empty')
        parts = ip_str.split(':')
        _min_parts = 3
        if len(parts) < _min_parts:
            msg = 'At least %d parts expected in %r' % (_min_parts, ip_str)
            raise AddressValueError(msg)
        if '.' in parts[-1]:
            try:
                ipv4_int = IPv4Address(parts.pop())._ip
            except AddressValueError as exc:
                raise AddressValueError('%s in %r' % (exc, ip_str)) from None
            parts.append('%x' % (ipv4_int >> 16 & 65535))
            parts.append('%x' % (ipv4_int & 65535))
        _max_parts = cls._HEXTET_COUNT + 1
        if len(parts) > _max_parts:
            msg = 'At most %d colons permitted in %r' % (_max_parts - 1, ip_str)
            raise AddressValueError(msg)
        skip_index = None
        for i in range(1, len(parts) - 1):
            if not parts[i]:
                if skip_index is not None:
                    msg = "At most one '::' permitted in %r" % ip_str
                    raise AddressValueError(msg)
                skip_index = i
        if skip_index is not None:
            parts_hi = skip_index
            parts_lo = len(parts) - skip_index - 1
            if not parts[0]:
                parts_hi -= 1
                if parts_hi:
                    msg = "Leading ':' only permitted as part of '::' in %r"
                    raise AddressValueError(msg % ip_str)
            if not parts[-1]:
                parts_lo -= 1
                if parts_lo:
                    msg = "Trailing ':' only permitted as part of '::' in %r"
                    raise AddressValueError(msg % ip_str)
            parts_skipped = cls._HEXTET_COUNT - (parts_hi + parts_lo)
            if parts_skipped < 1:
                msg = "Expected at most %d other parts with '::' in %r"
                raise AddressValueError(msg % (cls._HEXTET_COUNT - 1, ip_str))
        else:
            if len(parts) != cls._HEXTET_COUNT:
                msg = "Exactly %d parts expected without '::' in %r"
                raise AddressValueError(msg % (cls._HEXTET_COUNT, ip_str))
            if not parts[0]:
                msg = "Leading ':' only permitted as part of '::' in %r"
                raise AddressValueError(msg % ip_str)
            if not parts[-1]:
                msg = "Trailing ':' only permitted as part of '::' in %r"
                raise AddressValueError(msg % ip_str)
            parts_hi = len(parts)
            parts_lo = 0
            parts_skipped = 0
        try:
            ip_int = 0
            for i in range(parts_hi):
                ip_int <<= 16
                ip_int |= cls._parse_hextet(parts[i])
            ip_int <<= 16 * parts_skipped
            for i in range(-parts_lo, 0):
                ip_int <<= 16
                ip_int |= cls._parse_hextet(parts[i])
            return ip_int
        except ValueError as exc:
            raise AddressValueError('%s in %r' % (exc, ip_str)) from None

    @classmethod
    def _parse_hextet(cls, hextet_str):
        """Convert an IPv6 hextet string into an integer.

        Args:
            hextet_str: A string, the number to parse.

        Returns:
            The hextet as an integer.

        Raises:
            ValueError: if the input isn't strictly a hex number from
              [0..FFFF].

        """
        if not cls._HEX_DIGITS.issuperset(hextet_str):
            raise ValueError('Only hex digits permitted in %r' % hextet_str)
        if len(hextet_str) > 4:
            msg = 'At most 4 characters permitted in %r'
            raise ValueError(msg % hextet_str)
        return int(hextet_str, 16)

    @classmethod
    def _compress_hextets(cls, hextets):
        """Compresses a list of hextets.

        Compresses a list of strings, replacing the longest continuous
        sequence of "0" in the list with "" and adding empty strings at
        the beginning or at the end of the string such that subsequently
        calling ":".join(hextets) will produce the compressed version of
        the IPv6 address.

        Args:
            hextets: A list of strings, the hextets to compress.

        Returns:
            A list of strings.

        """
        best_doublecolon_start = -1
        best_doublecolon_len = 0
        doublecolon_start = -1
        doublecolon_len = 0
        for index, hextet in enumerate(hextets):
            if hextet == '0':
                doublecolon_len += 1
                if doublecolon_start == -1:
                    doublecolon_start = index
                if doublecolon_len > best_doublecolon_len:
                    best_doublecolon_len = doublecolon_len
                    best_doublecolon_start = doublecolon_start
            else:
                doublecolon_len = 0
                doublecolon_start = -1
        if best_doublecolon_len > 1:
            best_doublecolon_end = best_doublecolon_start + best_doublecolon_len
            if best_doublecolon_end == len(hextets):
                hextets += ['']
            hextets[best_doublecolon_start:best_doublecolon_end] = ['']
            if best_doublecolon_start == 0:
                hextets = [''] + hextets
        return hextets

    @classmethod
    def _string_from_ip_int(cls, ip_int=None):
        """Turns a 128-bit integer into hexadecimal notation.

        Args:
            ip_int: An integer, the IP address.

        Returns:
            A string, the hexadecimal representation of the address.

        Raises:
            ValueError: The address is bigger than 128 bits of all ones.

        """
        if ip_int is None:
            ip_int = int(cls._ip)
        if ip_int > cls._ALL_ONES:
            raise ValueError('IPv6 address is too large')
        hex_str = '%032x' % ip_int
        hextets = ['%x' % int(hex_str[x:x + 4], 16) for x in range(0, 32, 4)]
        hextets = cls._compress_hextets(hextets)
        return ':'.join(hextets)

    def _explode_shorthand_ip_string(self):
        """Expand a shortened IPv6 address.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            A string, the expanded IPv6 address.

        """
        if isinstance(self, IPv6Network):
            ip_str = str(self.network_address)
        elif isinstance(self, IPv6Interface):
            ip_str = str(self.ip)
        else:
            ip_str = str(self)
        ip_int = self._ip_int_from_string(ip_str)
        hex_str = '%032x' % ip_int
        parts = [hex_str[x:x + 4] for x in range(0, 32, 4)]
        if isinstance(self, (_BaseNetwork, IPv6Interface)):
            return '%s/%d' % (':'.join(parts), self._prefixlen)
        return ':'.join(parts)

    def _reverse_pointer(self):
        """Return the reverse DNS pointer name for the IPv6 address.

        This implements the method described in RFC3596 2.5.

        """
        reverse_chars = self.exploded[::-1].replace(':', '')
        return '.'.join(reverse_chars) + '.ip6.arpa'

    @staticmethod
    def _split_scope_id(ip_str):
        """Helper function to parse IPv6 string address with scope id.

        See RFC 4007 for details.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            (addr, scope_id) tuple.

        """
        addr, sep, scope_id = ip_str.partition('%')
        if not sep:
            scope_id = None
        elif not scope_id or '%' in scope_id:
            raise AddressValueError('Invalid IPv6 address: "%r"' % ip_str)
        return (addr, scope_id)

    @property
    def max_prefixlen(self):
        return self._max_prefixlen

    @property
    def version(self):
        return self._version