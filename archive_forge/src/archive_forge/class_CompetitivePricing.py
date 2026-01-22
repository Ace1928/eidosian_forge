from decimal import Decimal
from boto.compat import filter, map
class CompetitivePricing(ResponseElement):
    CompetitivePrices = Element(CompetitivePriceList)
    NumberOfOfferListings = SimpleList()
    TradeInValue = Element(ComplexMoney)