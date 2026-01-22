import ipaddress
from abc import ABCMeta
from typing import Any, cast, Dict, List, Optional, Union
import geoip2.records
from geoip2.mixins import SimpleEquality
class AnonymousIP(SimpleModel):
    """Model class for the GeoIP2 Anonymous IP.

    This class provides the following attribute:

    .. attribute:: is_anonymous

      This is true if the IP address belongs to any sort of anonymous network.

      :type: bool

    .. attribute:: is_anonymous_vpn

      This is true if the IP address is registered to an anonymous VPN
      provider.

      If a VPN provider does not register subnets under names associated with
      them, we will likely only flag their IP ranges using the
      ``is_hosting_provider`` attribute.

      :type: bool

    .. attribute:: is_hosting_provider

      This is true if the IP address belongs to a hosting or VPN provider
      (see description of ``is_anonymous_vpn`` attribute).

      :type: bool

    .. attribute:: is_public_proxy

      This is true if the IP address belongs to a public proxy.

      :type: bool

    .. attribute:: is_residential_proxy

      This is true if the IP address is on a suspected anonymizing network
      and belongs to a residential ISP.

      :type: bool

    .. attribute:: is_tor_exit_node

      This is true if the IP address is a Tor exit node.

      :type: bool

    .. attribute:: ip_address

      The IP address used in the lookup.

      :type: str

    .. attribute:: network

      The network associated with the record. In particular, this is the
      largest network where all of the fields besides ip_address have the same
      value.

      :type: ipaddress.IPv4Network or ipaddress.IPv6Network
    """
    is_anonymous: bool
    is_anonymous_vpn: bool
    is_hosting_provider: bool
    is_public_proxy: bool
    is_residential_proxy: bool
    is_tor_exit_node: bool

    def __init__(self, raw: Dict[str, bool]) -> None:
        super().__init__(raw)
        self.is_anonymous = raw.get('is_anonymous', False)
        self.is_anonymous_vpn = raw.get('is_anonymous_vpn', False)
        self.is_hosting_provider = raw.get('is_hosting_provider', False)
        self.is_public_proxy = raw.get('is_public_proxy', False)
        self.is_residential_proxy = raw.get('is_residential_proxy', False)
        self.is_tor_exit_node = raw.get('is_tor_exit_node', False)