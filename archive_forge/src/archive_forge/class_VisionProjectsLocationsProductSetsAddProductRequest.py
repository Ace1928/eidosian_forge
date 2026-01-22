from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsAddProductRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsAddProductRequest object.

  Fields:
    addProductToProductSetRequest: A AddProductToProductSetRequest resource to
      be passed as the request body.
    name: Required. The resource name for the ProductSet to modify. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/productSets/PRODUCT_SET_ID`
  """
    addProductToProductSetRequest = _messages.MessageField('AddProductToProductSetRequest', 1)
    name = _messages.StringField(2, required=True)