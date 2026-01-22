from decimal import Decimal
from boto.compat import filter, map
class ListCartsResult(ResponseElement):
    CartList = Element(Cart=ElementList(Cart))