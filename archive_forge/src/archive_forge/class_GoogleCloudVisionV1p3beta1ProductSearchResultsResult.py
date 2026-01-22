from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1ProductSearchResultsResult(_messages.Message):
    """Information about a product.

  Fields:
    image: The resource name of the image from the product that is the closest
      match to the query.
    product: The Product.
    score: A confidence level on the match, ranging from 0 (no confidence) to
      1 (full confidence).
  """
    image = _messages.StringField(1)
    product = _messages.MessageField('GoogleCloudVisionV1p3beta1Product', 2)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)