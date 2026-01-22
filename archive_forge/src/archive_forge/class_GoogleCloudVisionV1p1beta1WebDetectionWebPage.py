from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1WebDetectionWebPage(_messages.Message):
    """Metadata for web pages.

  Fields:
    fullMatchingImages: Fully matching images on the page. Can include resized
      copies of the query image.
    pageTitle: Title for the web page, may contain HTML markups.
    partialMatchingImages: Partial matching images on the page. Those images
      are similar enough to share some key-point features. For example an
      original image will likely have partial matching for its crops.
    score: (Deprecated) Overall relevancy score for the web page.
    url: The result web page URL.
  """
    fullMatchingImages = _messages.MessageField('GoogleCloudVisionV1p1beta1WebDetectionWebImage', 1, repeated=True)
    pageTitle = _messages.StringField(2)
    partialMatchingImages = _messages.MessageField('GoogleCloudVisionV1p1beta1WebDetectionWebImage', 3, repeated=True)
    score = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    url = _messages.StringField(5)