from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1LineItem(_messages.Message):
    """A single item within an order.

  Fields:
    changeHistory: Output only. Changes made on the item that are not pending
      anymore which might be because they already took effect, were reverted
      by the customer, or were rejected by the partner. No more operations are
      allowed on these changes.
    lineItemId: Output only. Line item ID.
    lineItemInfo: Output only. Current state and information of this item. It
      tells what, e.g. which offer, is currently effective.
    pendingChange: Output only. A change made on the item which is pending and
      not yet effective. Absence of this field indicates the line item is not
      undergoing a change.
  """
    changeHistory = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemChange', 1, repeated=True)
    lineItemId = _messages.StringField(2)
    lineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 3)
    pendingChange = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemChange', 4)