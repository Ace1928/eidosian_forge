from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevenueShareTypeValueValuesEnum(_messages.Enum):
    """Method used to calculate the revenue that is shared with developers.

    Values:
      REVENUE_SHARE_TYPE_UNSPECIFIED: Revenue share type is not specified.
      FIXED: Fixed percentage of the total revenue will be shared. The
        percentage to be shared can be configured by the API provider.
      VOLUME_BANDED: Amount of revenue shared depends on the number of API
        calls. The API call volume ranges and the revenue share percentage for
        each volume can be configured by the API provider. **Note**: Not
        supported by Apigee at this time.
    """
    REVENUE_SHARE_TYPE_UNSPECIFIED = 0
    FIXED = 1
    VOLUME_BANDED = 2