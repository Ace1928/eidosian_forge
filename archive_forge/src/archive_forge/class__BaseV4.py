import functools
class _BaseV4:
    """Base IPv4 object.

    The following methods are used by IPv4 objects in both single IP
    addresses and networks.

    """
    __slots__ = ()
    _version = 4
    _ALL_ONES = 2 ** IPV4LENGTH - 1
    _max_prefixlen = IPV4LENGTH
    _netmask_cache = {}

    def _explode_shorthand_ip_string(self):
        return str(self)

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
                try:
                    prefixlen = cls._prefix_from_prefix_string(arg)
                except NetmaskValueError:
                    prefixlen = cls._prefix_from_ip_string(arg)
            netmask = IPv4Address(cls._ip_int_from_prefix(prefixlen))
            cls._netmask_cache[arg] = (netmask, prefixlen)
        return cls._netmask_cache[arg]

    @classmethod
    def _ip_int_from_string(cls, ip_str):
        """Turn the given IP string into an integer for comparison.

        Args:
            ip_str: A string, the IP ip_str.

        Returns:
            The IP ip_str as an integer.

        Raises:
            AddressValueError: if ip_str isn't a valid IPv4 Address.

        """
        if not ip_str:
            raise AddressValueError('Address cannot be empty')
        octets = ip_str.split('.')
        if len(octets) != 4:
            raise AddressValueError('Expected 4 octets in %r' % ip_str)
        try:
            return int.from_bytes(map(cls._parse_octet, octets), 'big')
        except ValueError as exc:
            raise AddressValueError('%s in %r' % (exc, ip_str)) from None

    @classmethod
    def _parse_octet(cls, octet_str):
        """Convert a decimal octet into an integer.

        Args:
            octet_str: A string, the number to parse.

        Returns:
            The octet as an integer.

        Raises:
            ValueError: if the octet isn't strictly a decimal from [0..255].

        """
        if not octet_str:
            raise ValueError('Empty octet not permitted')
        if not (octet_str.isascii() and octet_str.isdigit()):
            msg = 'Only decimal digits permitted in %r'
            raise ValueError(msg % octet_str)
        if len(octet_str) > 3:
            msg = 'At most 3 characters permitted in %r'
            raise ValueError(msg % octet_str)
        if octet_str != '0' and octet_str[0] == '0':
            msg = 'Leading zeros are not permitted in %r'
            raise ValueError(msg % octet_str)
        octet_int = int(octet_str, 10)
        if octet_int > 255:
            raise ValueError('Octet %d (> 255) not permitted' % octet_int)
        return octet_int

    @classmethod
    def _string_from_ip_int(cls, ip_int):
        """Turns a 32-bit integer into dotted decimal notation.

        Args:
            ip_int: An integer, the IP address.

        Returns:
            The IP address as a string in dotted decimal notation.

        """
        return '.'.join(map(str, ip_int.to_bytes(4, 'big')))

    def _reverse_pointer(self):
        """Return the reverse DNS pointer name for the IPv4 address.

        This implements the method described in RFC1035 3.5.

        """
        reverse_octets = str(self).split('.')[::-1]
        return '.'.join(reverse_octets) + '.in-addr.arpa'

    @property
    def max_prefixlen(self):
        return self._max_prefixlen

    @property
    def version(self):
        return self._version