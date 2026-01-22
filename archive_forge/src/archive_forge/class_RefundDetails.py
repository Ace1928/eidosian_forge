from decimal import Decimal
from boto.compat import filter, map
class RefundDetails(ResponseElement):
    RefundAmount = Element(ComplexMoney)
    FeeRefunded = Element(ComplexMoney)
    RefundStatus = Element()