import functools
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