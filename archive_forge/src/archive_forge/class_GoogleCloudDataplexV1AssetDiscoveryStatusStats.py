from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetDiscoveryStatusStats(_messages.Message):
    """The aggregated data statistics for the asset reported by discovery.

  Fields:
    dataItems: The count of data items within the referenced resource.
    dataSize: The number of stored data bytes within the referenced resource.
    filesets: The count of fileset entities within the referenced resource.
    tables: The count of table entities within the referenced resource.
  """
    dataItems = _messages.IntegerField(1)
    dataSize = _messages.IntegerField(2)
    filesets = _messages.IntegerField(3)
    tables = _messages.IntegerField(4)