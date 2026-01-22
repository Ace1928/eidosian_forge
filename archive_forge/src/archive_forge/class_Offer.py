from decimal import Decimal
from boto.compat import filter, map
class Offer(ResponseElement):
    BuyingPrice = Element(Price)
    RegularPrice = Element(ComplexMoney)