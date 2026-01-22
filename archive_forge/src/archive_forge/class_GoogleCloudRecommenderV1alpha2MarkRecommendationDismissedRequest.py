from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest(_messages.Message):
    """Request for the `MarkRecommendationDismissed` Method.

  Fields:
    etag: Fingerprint of the Recommendation. Provides optimistic locking.
  """
    etag = _messages.StringField(1)