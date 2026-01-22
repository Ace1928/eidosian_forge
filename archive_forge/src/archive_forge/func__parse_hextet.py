import functools
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