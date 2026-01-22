import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class Traits(Record):
    """Contains data for the traits record associated with an IP address.

    This class contains the traits data associated with an IP address.

    This class has the following attributes:


    .. attribute:: autonomous_system_number

      The `autonomous system
      number <https://en.wikipedia.org/wiki/Autonomous_system_(Internet)>`_
      associated with the IP address. This attribute is only available from
      the City Plus and Insights web services and the Enterprise database.

      :type: int

    .. attribute:: autonomous_system_organization

      The organization associated with the registered `autonomous system
      number <https://en.wikipedia.org/wiki/Autonomous_system_(Internet)>`_ for
      the IP address. This attribute is only available from the City Plus and
      Insights web service end points and the Enterprise database.

      :type: str

    .. attribute:: connection_type

      The connection type may take the following values:

      - Dialup
      - Cable/DSL
      - Corporate
      - Cellular
      - Satellite

      Additional values may be added in the future.

      This attribute is only available from the City Plus and Insights web
      service end points and the Enterprise database.

      :type: str

    .. attribute:: domain

      The second level domain associated with the
      IP address. This will be something like "example.com" or
      "example.co.uk", not "foo.example.com". This attribute is only available
      from the City Plus and Insights web service end points and the
      Enterprise database.

      :type: str

    .. attribute:: ip_address

      The IP address that the data in the model
      is for. If you performed a "me" lookup against the web service, this
      will be the externally routable IP address for the system the code is
      running on. If the system is behind a NAT, this may differ from the IP
      address locally assigned to it.

      :type: str

    .. attribute:: is_anonymous

      This is true if the IP address belongs to any sort of anonymous network.
      This attribute is only available from Insights.

      :type: bool

    .. attribute:: is_anonymous_proxy

      This is true if the IP is an anonymous proxy.

      :type: bool

      .. deprecated:: 2.2.0
        Use our our `GeoIP2 Anonymous IP database
        <https://www.maxmind.com/en/geoip2-anonymous-ip-database GeoIP2>`_
        instead.

    .. attribute:: is_anonymous_vpn

      This is true if the IP address is registered to an anonymous VPN
      provider.

      If a VPN provider does not register subnets under names associated with
      them, we will likely only flag their IP ranges using the
      ``is_hosting_provider`` attribute.

      This attribute is only available from Insights.

      :type: bool

    .. attribute:: is_anycast

      This returns true if the IP address belongs to an
      `anycast network <https://en.wikipedia.org/wiki/Anycast>`_.
      This is available for the GeoIP2 Country, City Plus, and Insights
      web services and the GeoIP2 Country, City, and Enterprise databases.

      :type: bool

    .. attribute:: is_hosting_provider

      This is true if the IP address belongs to a hosting or VPN provider
      (see description of ``is_anonymous_vpn`` attribute).
      This attribute is only available from Insights.

      :type: bool

    .. attribute:: is_legitimate_proxy

      This attribute is true if MaxMind believes this IP address to be a
      legitimate proxy, such as an internal VPN used by a corporation. This
      attribute is only available in the Enterprise database.

      :type: bool

    .. attribute:: is_public_proxy

      This is true if the IP address belongs to a public proxy. This attribute
      is only available from Insights.

      :type: bool

    .. attribute:: is_residential_proxy

      This is true if the IP address is on a suspected anonymizing network
      and belongs to a residential ISP. This attribute is only available from
      Insights.

      :type: bool


    .. attribute:: is_satellite_provider

      This is true if the IP address is from a satellite provider that
      provides service to multiple countries.

      :type: bool

      .. deprecated:: 2.2.0
        Due to the increased coverage by mobile carriers, very few
        satellite providers now serve multiple countries. As a result, the
        output does not provide sufficiently relevant data for us to maintain
        it.

    .. attribute:: is_tor_exit_node

      This is true if the IP address is a Tor exit node. This attribute is
      only available from Insights.

      :type: bool

    .. attribute:: isp

      The name of the ISP associated with the IP address. This attribute is
      only available from the City Plus and Insights web services and the
      Enterprise database.

      :type: str

    .. attribute: mobile_country_code

      The `mobile country code (MCC)
      <https://en.wikipedia.org/wiki/Mobile_country_code>`_ associated with the
      IP address and ISP. This attribute is available from the City Plus and
      Insights web services and the Enterprise database.

      :type: str

    .. attribute: mobile_network_code

      The `mobile network code (MNC)
      <https://en.wikipedia.org/wiki/Mobile_country_code>`_ associated with the
      IP address and ISP. This attribute is available from the City Plus and
      Insights web services and the Enterprise database.

      :type: str

    .. attribute:: network

      The network associated with the record. In particular, this is the
      largest network where all of the fields besides ip_address have the same
      value.

      :type: ipaddress.IPv4Network or ipaddress.IPv6Network

    .. attribute:: organization

      The name of the organization associated with the IP address. This
      attribute is only available from the City Plus and Insights web services
      and the Enterprise database.

      :type: str

    .. attribute:: static_ip_score

      An indicator of how static or dynamic an IP address is. The value ranges
      from 0 to 99.99 with higher values meaning a greater static association.
      For example, many IP addresses with a user_type of cellular have a
      lifetime under one. Static Cable/DSL IPs typically have a lifetime above
      thirty.

      This indicator can be useful for deciding whether an IP address represents
      the same user over time. This attribute is only available from
      Insights.

      :type: float

    .. attribute:: user_count

      The estimated number of users sharing the IP/network during the past 24
      hours. For IPv4, the count is for the individual IP. For IPv6, the count
      is for the /64 network. This attribute is only available from
      Insights.

      :type: int

    .. attribute:: user_type

      The user type associated with the IP
      address. This can be one of the following values:

      * business
      * cafe
      * cellular
      * college
      * consumer_privacy_network
      * content_delivery_network
      * dialup
      * government
      * hosting
      * library
      * military
      * residential
      * router
      * school
      * search_engine_spider
      * traveler

      This attribute is only available from the Insights end point and the
      Enterprise database.

      :type: str

    """
    autonomous_system_number: Optional[int]
    autonomous_system_organization: Optional[str]
    connection_type: Optional[str]
    domain: Optional[str]
    ip_address: Optional[str]
    is_anonymous: bool
    is_anonymous_proxy: bool
    is_anonymous_vpn: bool
    is_anycast: bool
    is_hosting_provider: bool
    is_legitimate_proxy: bool
    is_public_proxy: bool
    is_residential_proxy: bool
    is_satellite_provider: bool
    is_tor_exit_node: bool
    isp: Optional[str]
    mobile_country_code: Optional[str]
    mobile_network_code: Optional[str]
    organization: Optional[str]
    static_ip_score: Optional[float]
    user_count: Optional[int]
    user_type: Optional[str]
    _network: Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]
    _prefix_len: Optional[int]

    def __init__(self, autonomous_system_number: Optional[int]=None, autonomous_system_organization: Optional[str]=None, connection_type: Optional[str]=None, domain: Optional[str]=None, is_anonymous: bool=False, is_anonymous_proxy: bool=False, is_anonymous_vpn: bool=False, is_hosting_provider: bool=False, is_legitimate_proxy: bool=False, is_public_proxy: bool=False, is_residential_proxy: bool=False, is_satellite_provider: bool=False, is_tor_exit_node: bool=False, isp: Optional[str]=None, ip_address: Optional[str]=None, network: Optional[str]=None, organization: Optional[str]=None, prefix_len: Optional[int]=None, static_ip_score: Optional[float]=None, user_count: Optional[int]=None, user_type: Optional[str]=None, mobile_country_code: Optional[str]=None, mobile_network_code: Optional[str]=None, is_anycast: bool=False, **_) -> None:
        self.autonomous_system_number = autonomous_system_number
        self.autonomous_system_organization = autonomous_system_organization
        self.connection_type = connection_type
        self.domain = domain
        self.is_anonymous = is_anonymous
        self.is_anonymous_proxy = is_anonymous_proxy
        self.is_anonymous_vpn = is_anonymous_vpn
        self.is_anycast = is_anycast
        self.is_hosting_provider = is_hosting_provider
        self.is_legitimate_proxy = is_legitimate_proxy
        self.is_public_proxy = is_public_proxy
        self.is_residential_proxy = is_residential_proxy
        self.is_satellite_provider = is_satellite_provider
        self.is_tor_exit_node = is_tor_exit_node
        self.isp = isp
        self.mobile_country_code = mobile_country_code
        self.mobile_network_code = mobile_network_code
        self.organization = organization
        self.static_ip_score = static_ip_score
        self.user_type = user_type
        self.user_count = user_count
        self.ip_address = ip_address
        if network is None:
            self._network = None
        else:
            self._network = ipaddress.ip_network(network, False)
        self._prefix_len = prefix_len

    @property
    def network(self) -> Optional[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
        """The network for the record"""
        network = self._network
        if network is not None:
            return network
        ip_address = self.ip_address
        prefix_len = self._prefix_len
        if ip_address is None or prefix_len is None:
            return None
        network = ipaddress.ip_network(f'{ip_address}/{prefix_len}', False)
        self._network = network
        return network