import functools
@classmethod
def _prefix_from_prefix_string(cls, prefixlen_str):
    """Return prefix length from a numeric string

        Args:
            prefixlen_str: The string to be converted

        Returns:
            An integer, the prefix length.

        Raises:
            NetmaskValueError: If the input is not a valid netmask
        """
    if not (prefixlen_str.isascii() and prefixlen_str.isdigit()):
        cls._report_invalid_netmask(prefixlen_str)
    try:
        prefixlen = int(prefixlen_str)
    except ValueError:
        cls._report_invalid_netmask(prefixlen_str)
    if not 0 <= prefixlen <= cls._max_prefixlen:
        cls._report_invalid_netmask(prefixlen_str)
    return prefixlen