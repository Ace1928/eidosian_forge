from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3BucketScope(_messages.Message):
    """Configuration to scope a crawler to the provided list of buckets.

  Fields:
    buckets: The maximum number of buckets allowed is 1000.
  """
    buckets = _messages.MessageField('GoogleCloudDatacatalogV1alpha3BucketSpec', 1, repeated=True)