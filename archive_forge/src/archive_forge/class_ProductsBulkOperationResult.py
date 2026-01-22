from decimal import Decimal
from boto.compat import filter, map
class ProductsBulkOperationResult(ResponseElement):
    Product = Element(Product)
    Error = Element()