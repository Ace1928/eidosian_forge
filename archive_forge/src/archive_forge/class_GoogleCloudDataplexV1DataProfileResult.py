from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileResult(_messages.Message):
    """DataProfileResult defines the output of DataProfileScan. Each field of
  the table will have field type specific profile result.

  Fields:
    postScanActionsResult: Output only. The result of post scan actions.
    profile: The profile information per field.
    rowCount: The count of rows scanned.
    scannedData: The data scanned for this result.
  """
    postScanActionsResult = _messages.MessageField('GoogleCloudDataplexV1DataProfileResultPostScanActionsResult', 1)
    profile = _messages.MessageField('GoogleCloudDataplexV1DataProfileResultProfile', 2)
    rowCount = _messages.IntegerField(3)
    scannedData = _messages.MessageField('GoogleCloudDataplexV1ScannedData', 4)