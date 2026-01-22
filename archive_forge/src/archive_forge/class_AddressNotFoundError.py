import ipaddress
from typing import Optional, Union
class AddressNotFoundError(GeoIP2Error):
    """The address you were looking up was not found.

    .. attribute:: ip_address

      The IP address used in the lookup. This is only available for database
      lookups.

      :type: str

    .. attribute:: network

      The network associated with the error. In particular, this is the
      largest network where no address would be found. This is only
      available for database lookups.

      :type: ipaddress.IPv4Network or ipaddress.IPv6Network

    """
    ip_address: Optional[str]
    _prefix_len: Optional[int]

    def __init__(self, message: str, ip_address: Optional[str]=None, prefix_len: Optional[int]=None) -> None:
        super().__init__(message)
        self.ip_address = ip_address
        self._prefix_len = prefix_len

    @property
    def network(self) -> Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
        """The network for the error"""
        if self.ip_address is None or self._prefix_len is None:
            return None
        return ipaddress.ip_network(f'{self.ip_address}/{self._prefix_len}', False)