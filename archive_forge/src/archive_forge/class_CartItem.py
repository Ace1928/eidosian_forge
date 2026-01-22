from decimal import Decimal
from boto.compat import filter, map
class CartItem(ResponseElement):
    CurrentPrice = Element(ComplexMoney)
    SalePrice = Element(ComplexMoney)