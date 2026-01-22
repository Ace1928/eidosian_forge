from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TensorboardTimeSeriesMetadata(_messages.Message):
    """Describes metadata for a TensorboardTimeSeries.

  Fields:
    maxBlobSequenceLength: Output only. The largest blob sequence length
      (number of blobs) of all data points in this time series, if its
      ValueType is BLOB_SEQUENCE.
    maxStep: Output only. Max step index of all data points within a
      TensorboardTimeSeries.
    maxWallTime: Output only. Max wall clock timestamp of all data points
      within a TensorboardTimeSeries.
  """
    maxBlobSequenceLength = _messages.IntegerField(1)
    maxStep = _messages.IntegerField(2)
    maxWallTime = _messages.StringField(3)