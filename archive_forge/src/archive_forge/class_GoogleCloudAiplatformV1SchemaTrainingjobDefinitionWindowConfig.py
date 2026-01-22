from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionWindowConfig(_messages.Message):
    """Config that contains the strategy used to generate sliding windows in
  time series training. A window is a series of rows that comprise the context
  up to the time of prediction, and the horizon following. The corresponding
  row for each window marks the start of the forecast horizon. Each window is
  used as an input example for training/evaluation.

  Fields:
    column: Name of the column that should be used to generate sliding
      windows. The column should contain either booleans or string booleans;
      if the value of the row is True, generate a sliding window with the
      horizon starting at that row. The column will not be used as a feature
      in training.
    maxCount: Maximum number of windows that should be generated across all
      time series.
    strideLength: Stride length used to generate input examples. Within one
      time series, every {$STRIDE_LENGTH} rows will be used to generate a
      sliding window.
  """
    column = _messages.StringField(1)
    maxCount = _messages.IntegerField(2)
    strideLength = _messages.IntegerField(3)