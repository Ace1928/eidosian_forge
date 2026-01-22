from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductsReferenceImagesDeleteRequest(_messages.Message):
    """A VisionProjectsLocationsProductsReferenceImagesDeleteRequest object.

  Fields:
    name: Required. The resource name of the reference image to delete. Format
      is: `projects/PROJECT_ID/locations/LOC_ID/products/PRODUCT_ID/referenceI
      mages/IMAGE_ID`
  """
    name = _messages.StringField(1, required=True)