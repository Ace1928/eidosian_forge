from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvertisedGroupsValueListEntryValuesEnum(_messages.Enum):
    """AdvertisedGroupsValueListEntryValuesEnum enum type.

    Values:
      ALL_SUBNETS: Advertise all available subnets (including peer VPC
        subnets).
    """
    ALL_SUBNETS = 0