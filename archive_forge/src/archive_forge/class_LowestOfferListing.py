from decimal import Decimal
from boto.compat import filter, map
class LowestOfferListing(ResponseElement):
    Qualifiers = Element(ShippingTime=Element())
    Price = Element(Price)