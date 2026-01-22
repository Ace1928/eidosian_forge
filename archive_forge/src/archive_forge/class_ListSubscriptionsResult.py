from decimal import Decimal
from boto.compat import filter, map
class ListSubscriptionsResult(ResponseElement):
    SubscriptionList = MemberList(Subscription)