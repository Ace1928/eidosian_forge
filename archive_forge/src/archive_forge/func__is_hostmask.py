from __future__ import unicode_literals
import itertools
import struct
def _is_hostmask(self, ip_str):
    """Test if the IP string is a hostmask (rather than a netmask).

        Args:
            ip_str: A string, the potential hostmask.

        Returns:
            A boolean, True if the IP string is a hostmask.

        """
    bits = ip_str.split('.')
    try:
        parts = [x for x in map(int, bits) if x in self._valid_mask_octets]
    except ValueError:
        return False
    if len(parts) != len(bits):
        return False
    if parts[0] < parts[-1]:
        return True
    return False