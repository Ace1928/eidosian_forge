from decimal import Decimal
from boto.compat import filter, map
class ListRegisteredDestinationsResult(ResponseElement):
    DestinationList = MemberList(Destination)