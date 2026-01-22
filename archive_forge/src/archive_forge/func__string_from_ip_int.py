import functools
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