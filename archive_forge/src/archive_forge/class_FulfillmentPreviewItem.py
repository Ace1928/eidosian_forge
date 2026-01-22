from decimal import Decimal
from boto.compat import filter, map
class FulfillmentPreviewItem(ResponseElement):
    EstimatedShippingWeight = Element(ComplexWeight)