from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequest(_messages.Message):
    """Request message to update an order for non-quote products.

  Fields:
    modifications: A GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProdu
      ctsOrderRequestModification attribute.
  """
    modifications = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequestModification', 1, repeated=True)