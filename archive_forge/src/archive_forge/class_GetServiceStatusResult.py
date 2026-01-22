from decimal import Decimal
from boto.compat import filter, map
class GetServiceStatusResult(ResponseElement):
    Messages = Element(Messages=ElementList())