import functools
@classmethod
def _ip_int_from_prefix(cls, prefixlen):
    """Turn the prefix length into a bitwise netmask

        Args:
            prefixlen: An integer, the prefix length.

        Returns:
            An integer.

        """
    return cls._ALL_ONES ^ cls._ALL_ONES >> prefixlen