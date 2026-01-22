from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterNatSubnetworkToNat(_messages.Message):
    """Defines the IP ranges that want to use NAT for a subnetwork.

  Enums:
    SourceIpRangesToNatValueListEntryValuesEnum:

  Fields:
    name: URL for the subnetwork resource that will use NAT.
    secondaryIpRangeNames: A list of the secondary ranges of the Subnetwork
      that are allowed to use NAT. This can be populated only if
      "LIST_OF_SECONDARY_IP_RANGES" is one of the values in
      source_ip_ranges_to_nat.
    sourceIpRangesToNat: Specify the options for NAT ranges in the Subnetwork.
      All options of a single value are valid except
      NAT_IP_RANGE_OPTION_UNSPECIFIED. The only valid option with multiple
      values is: ["PRIMARY_IP_RANGE", "LIST_OF_SECONDARY_IP_RANGES"] Default:
      [ALL_IP_RANGES]
  """

    class SourceIpRangesToNatValueListEntryValuesEnum(_messages.Enum):
        """SourceIpRangesToNatValueListEntryValuesEnum enum type.

    Values:
      ALL_IP_RANGES: The primary and all the secondary ranges are allowed to
        Nat.
      LIST_OF_SECONDARY_IP_RANGES: A list of secondary ranges are allowed to
        Nat.
      PRIMARY_IP_RANGE: The primary range is allowed to Nat.
    """
        ALL_IP_RANGES = 0
        LIST_OF_SECONDARY_IP_RANGES = 1
        PRIMARY_IP_RANGE = 2
    name = _messages.StringField(1)
    secondaryIpRangeNames = _messages.StringField(2, repeated=True)
    sourceIpRangesToNat = _messages.EnumField('SourceIpRangesToNatValueListEntryValuesEnum', 3, repeated=True)