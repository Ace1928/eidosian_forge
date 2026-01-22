from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CommonUsageStats(_messages.Message):
    """Common statistics on the entry's usage. They can be set on any system.

  Fields:
    viewCount: View count in source system.
  """
    viewCount = _messages.IntegerField(1)