import functools
@classmethod
def _prefix_from_ip_int(cls, ip_int):
    """Return prefix length from the bitwise netmask.

        Args:
            ip_int: An integer, the netmask in expanded bitwise format

        Returns:
            An integer, the prefix length.

        Raises:
            ValueError: If the input intermingles zeroes & ones
        """
    trailing_zeroes = _count_righthand_zero_bits(ip_int, cls._max_prefixlen)
    prefixlen = cls._max_prefixlen - trailing_zeroes
    leading_ones = ip_int >> trailing_zeroes
    all_ones = (1 << prefixlen) - 1
    if leading_ones != all_ones:
        byteslen = cls._max_prefixlen // 8
        details = ip_int.to_bytes(byteslen, 'big')
        msg = 'Netmask pattern %r mixes zeroes & ones'
        raise ValueError(msg % details)
    return prefixlen