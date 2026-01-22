import ipaddress
from abc import ABCMeta
from typing import Any, cast, Dict, List, Optional, Union
import geoip2.records
from geoip2.mixins import SimpleEquality
class ISP(ASN):
    """Model class for the GeoIP2 ISP.

    This class provides the following attribute:

    .. attribute:: autonomous_system_number

      The autonomous system number associated with the IP address.

      :type: int

    .. attribute:: autonomous_system_organization

      The organization associated with the registered autonomous system number
      for the IP address.

      :type: str

    .. attribute:: isp

      The name of the ISP associated with the IP address.

      :type: str

    .. attribute: mobile_country_code

      The `mobile country code (MCC)
      <https://en.wikipedia.org/wiki/Mobile_country_code>`_ associated with the
      IP address and ISP.

      :type: str

    .. attribute: mobile_network_code

      The `mobile network code (MNC)
      <https://en.wikipedia.org/wiki/Mobile_country_code>`_ associated with the
      IP address and ISP.

      :type: str

    .. attribute:: organization

      The name of the organization associated with the IP address.

      :type: str

    .. attribute:: ip_address

      The IP address used in the lookup.

      :type: str

    .. attribute:: network

      The network associated with the record. In particular, this is the
      largest network where all of the fields besides ip_address have the same
      value.

      :type: ipaddress.IPv4Network or ipaddress.IPv6Network
    """
    isp: Optional[str]
    mobile_country_code: Optional[str]
    mobile_network_code: Optional[str]
    organization: Optional[str]

    def __init__(self, raw: Dict[str, Union[str, int]]) -> None:
        super().__init__(raw)
        self.isp = cast(Optional[str], raw.get('isp'))
        self.mobile_country_code = cast(Optional[str], raw.get('mobile_country_code'))
        self.mobile_network_code = cast(Optional[str], raw.get('mobile_network_code'))
        self.organization = cast(Optional[str], raw.get('organization'))