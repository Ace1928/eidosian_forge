from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubscriptionPlanValueValuesEnum(_messages.Enum):
    """Output only. Subscription plan that the customer has purchased. Output
    only.

    Values:
      SUBSCRIPTION_PLAN_UNSPECIFIED: Subscription plan not specified.
      SUBSCRIPTION_2021: Traditional subscription plan.
      SUBSCRIPTION_2024: New subscription plan that provides standard proxy
        and scaled proxy implementation.
    """
    SUBSCRIPTION_PLAN_UNSPECIFIED = 0
    SUBSCRIPTION_2021 = 1
    SUBSCRIPTION_2024 = 2