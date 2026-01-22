from decimal import Decimal
from boto.compat import filter, map
class SetOrderReferenceDetailsResult(ResponseElement):
    OrderReferenceDetails = Element(OrderReferenceDetails)