from decimal import Decimal
from boto.compat import filter, map
class GetFulfillmentOrderResult(ResponseElement):
    FulfillmentOrder = Element(FulfillmentOrder)
    FulfillmentShipment = MemberList(FulfillmentShipmentItem=MemberList(), FulfillmentShipmentPackage=MemberList())
    FulfillmentOrderItem = MemberList()