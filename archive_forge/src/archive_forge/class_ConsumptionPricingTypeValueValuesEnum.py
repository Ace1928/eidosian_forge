from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumptionPricingTypeValueValuesEnum(_messages.Enum):
    """Pricing model used for consumption-based charges.

    Values:
      CONSUMPTION_PRICING_TYPE_UNSPECIFIED: Pricing model not specified. This
        is the default.
      FIXED_PER_UNIT: Fixed rate charged for each API call.
      BANDED: Variable rate charged for each API call based on price tiers.
        Example: * 1-100 calls cost $2 per call * 101-200 calls cost $1.50 per
        call * 201-300 calls cost $1 per call * Total price for 50 calls: 50 x
        $2 = $100 * Total price for 150 calls: 100 x $2 + 50 x $1.5 = $275 *
        Total price for 250 calls: 100 x $2 + 100 x $1.5 + 50 x $1 = $400.
        **Note**: Not supported by Apigee at this time.
      TIERED: **Note**: Not supported by Apigee at this time.
      STAIRSTEP: **Note**: Not supported by Apigee at this time.
      BUNDLES: Cumulative rate charged for bundle of API calls whether or not
        the entire bundle is used. Example: * 1-100 calls cost $150 flat fee.
        * 101-200 calls cost $100 flat free. * 201-300 calls cost $75 flat
        fee. * Total price for 1 call: $150 * Total price for 50 calls: $150 *
        Total price for 150 calls: $150 + $100 * Total price for 250 calls:
        $150 + $100 + $75
    """
    CONSUMPTION_PRICING_TYPE_UNSPECIFIED = 0
    FIXED_PER_UNIT = 1
    BANDED = 2
    TIERED = 3
    STAIRSTEP = 4
    BUNDLES = 5