from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1ReferenceImage(_messages.Message):
    """A `ReferenceImage` represents a product image and its associated
  metadata, such as bounding boxes.

  Fields:
    boundingPolys: Optional. Bounding polygons around the areas of interest in
      the reference image. If this field is empty, the system will try to
      detect regions of interest. At most 10 bounding polygons will be used.
      The provided shape is converted into a non-rotated rectangle. Once
      converted, the small edge of the rectangle must be greater than or equal
      to 300 pixels. The aspect ratio must be 1:4 or less (i.e. 1:3 is ok; 1:5
      is not).
    name: The resource name of the reference image. Format is: `projects/PROJE
      CT_ID/locations/LOC_ID/products/PRODUCT_ID/referenceImages/IMAGE_ID`.
      This field is ignored when creating a reference image.
    uri: Required. The Google Cloud Storage URI of the reference image. The
      URI must start with `gs://`.
  """
    boundingPolys = _messages.MessageField('GoogleCloudVisionV1p3beta1BoundingPoly', 1, repeated=True)
    name = _messages.StringField(2)
    uri = _messages.StringField(3)