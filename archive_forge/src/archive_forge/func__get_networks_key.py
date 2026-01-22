import functools
def _get_networks_key(self):
    """Network-only key function.

        Returns an object that identifies this address' network and
        netmask. This function is a suitable "key" argument for sorted()
        and list.sort().

        """
    return (self._version, self.network_address, self.netmask)