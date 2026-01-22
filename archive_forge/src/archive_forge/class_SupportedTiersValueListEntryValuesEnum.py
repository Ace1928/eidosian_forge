from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedTiersValueListEntryValuesEnum(_messages.Enum):
    """SupportedTiersValueListEntryValuesEnum enum type.

    Values:
      SERVICE_TIER_UNSPECIFIED: An invalid sentinel value, used to indicate
        that a tier has not been provided explicitly.
      SERVICE_TIER_BASIC: The Cloud Monitoring Basic tier, a free tier of
        service that provides basic features, a moderate allotment of logs,
        and access to built-in metrics. A number of features are not available
        in this tier. For more details, see the service tiers documentation
        (https://cloud.google.com/monitoring/workspaces/tiers).
      SERVICE_TIER_PREMIUM: The Cloud Monitoring Premium tier, a higher, more
        expensive tier of service that provides access to all Cloud Monitoring
        features, lets you use Cloud Monitoring with AWS accounts, and has a
        larger allotments for logs and metrics. For more details, see the
        service tiers documentation
        (https://cloud.google.com/monitoring/workspaces/tiers).
    """
    SERVICE_TIER_UNSPECIFIED = 0
    SERVICE_TIER_BASIC = 1
    SERVICE_TIER_PREMIUM = 2