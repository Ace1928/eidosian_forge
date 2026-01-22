from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UptimeCheckIp(_messages.Message):
    """Contains the region, location, and list of IP addresses where checkers
  in the location run from.

  Enums:
    RegionValueValuesEnum: A broad region category in which the IP address is
      located.

  Fields:
    ipAddress: The IP address from which the Uptime check originates. This is
      a fully specified IP address (not an IP address range). Most IP
      addresses, as of this publication, are in IPv4 format; however, one
      should not rely on the IP addresses being in IPv4 format indefinitely,
      and should support interpreting this field in either IPv4 or IPv6
      format.
    location: A more specific location within the region that typically
      encodes a particular city/town/metro (and its containing state/province
      or country) within the broader umbrella region category.
    region: A broad region category in which the IP address is located.
  """

    class RegionValueValuesEnum(_messages.Enum):
        """A broad region category in which the IP address is located.

    Values:
      REGION_UNSPECIFIED: Default value if no region is specified. Will result
        in Uptime checks running from all regions.
      USA: Allows checks to run from locations within the United States of
        America.
      EUROPE: Allows checks to run from locations within the continent of
        Europe.
      SOUTH_AMERICA: Allows checks to run from locations within the continent
        of South America.
      ASIA_PACIFIC: Allows checks to run from locations within the Asia
        Pacific area (ex: Singapore).
      USA_OREGON: Allows checks to run from locations within the western
        United States of America
      USA_IOWA: Allows checks to run from locations within the central United
        States of America
      USA_VIRGINIA: Allows checks to run from locations within the eastern
        United States of America
    """
        REGION_UNSPECIFIED = 0
        USA = 1
        EUROPE = 2
        SOUTH_AMERICA = 3
        ASIA_PACIFIC = 4
        USA_OREGON = 5
        USA_IOWA = 6
        USA_VIRGINIA = 7
    ipAddress = _messages.StringField(1)
    location = _messages.StringField(2)
    region = _messages.EnumField('RegionValueValuesEnum', 3)