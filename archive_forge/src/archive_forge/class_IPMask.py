import json
import netaddr
import re
class IPMask(Decoder):
    """IPMask stores an IPv6 or IPv4 and a mask.

    It uses netaddr.IPAddress.

    IPMasks can represent valid CIDRs or randomly masked IP Addresses.

    Args:
        string (str): A string representing the ip/mask.
    """

    def __init__(self, string):
        self._ipnet = None
        self._ip = None
        self._mask = None
        try:
            self._ipnet = netaddr.IPNetwork(string)
        except netaddr.AddrFormatError:
            pass
        if not self._ipnet:
            parts = string.split('/')
            if len(parts) != 2:
                raise ValueError('value {}: is not an ipv4 or ipv6 address'.format(string))
            try:
                self._ip = netaddr.IPAddress(parts[0])
                self._mask = netaddr.IPAddress(parts[1])
            except netaddr.AddrFormatError as exc:
                raise ValueError('value {}: is not an ipv4 or ipv6 address'.format(string)) from exc

    def __eq__(self, other):
        """Equality operator.

        Both the IPAddress and the mask are compared. This can be used
        to implement filters where a specific mask is expected, e.g:
        nw_src=192.168.1.0/24.

        Args:
            other (IPMask or netaddr.IPNetwork or netaddr.IPAddress):
                Another IPAddress or IPNetwork to compare against.

        Returns:
            True if this IPMask is the same as the other.
        """
        if isinstance(other, netaddr.IPNetwork):
            return self._ipnet and self._ipnet == other
        if isinstance(other, netaddr.IPAddress):
            return self._ipnet and self._ipnet.ip == other
        elif isinstance(other, IPMask):
            if self._ipnet:
                return self._ipnet == other._ipnet
            return self._ip == other._ip and self._mask == other._mask
        else:
            return False

    def __contains__(self, other):
        """Contains operator.

        Only comparing valid CIDRs is supported.

        Args:
            other (netaddr.IPAddress or IPMask): An IP address.

        Returns:
            True if the other IPAddress is contained in this IPMask's address
            range.
        """
        if isinstance(other, IPMask):
            if not other._ipnet:
                raise ValueError('Only comparing valid CIDRs is supported')
            return netaddr.IPAddress(other._ipnet.first) in self and netaddr.IPAddress(other._ipnet.last) in self
        elif isinstance(other, netaddr.IPAddress):
            if self._ipnet:
                return other in self._ipnet
            return other & self._mask == self._ip & self._mask

    def cidr(self):
        """
        Returns True if the IPMask is a valid CIDR.
        """
        return self._ipnet is not None

    @property
    def ip(self):
        """The IP address."""
        if self._ipnet:
            return self._ipnet.ip
        return self._ip

    @property
    def mask(self):
        """The IP mask."""
        if self._ipnet:
            return self._ipnet.netmask
        return self._mask

    def __str__(self):
        if self._ipnet:
            return str(self._ipnet)
        return '/'.join([str(self._ip), str(self._mask)])

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self)

    def to_json(self):
        return str(self)