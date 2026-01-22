from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEventDataProfileResult(_messages.Message):
    """Data profile result for data scan job.

  Fields:
    rowCount: The count of rows processed in the data scan job.
  """
    rowCount = _messages.IntegerField(1)