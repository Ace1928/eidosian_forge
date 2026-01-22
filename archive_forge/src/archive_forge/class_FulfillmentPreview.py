from decimal import Decimal
from boto.compat import filter, map
class FulfillmentPreview(ResponseElement):
    EstimatedShippingWeight = Element(ComplexWeight)
    EstimatedFees = MemberList(Amount=Element(ComplexAmount))
    UnfulfillablePreviewItems = MemberList(FulfillmentPreviewItem)
    FulfillmentPreviewShipments = MemberList(FulfillmentPreviewItems=MemberList(FulfillmentPreviewItem))