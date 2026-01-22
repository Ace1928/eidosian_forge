from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TimeSeriesData(_messages.Message):
    """All the data stored in a TensorboardTimeSeries.

  Enums:
    ValueTypeValueValuesEnum: Required. Immutable. The value type of this time
      series. All the values in this time series data must match this value
      type.

  Fields:
    tensorboardTimeSeriesId: Required. The ID of the TensorboardTimeSeries,
      which will become the final component of the TensorboardTimeSeries'
      resource name
    valueType: Required. Immutable. The value type of this time series. All
      the values in this time series data must match this value type.
    values: Required. Data points in this time series.
  """

    class ValueTypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The value type of this time series. All the
    values in this time series data must match this value type.

    Values:
      VALUE_TYPE_UNSPECIFIED: The value type is unspecified.
      SCALAR: Used for TensorboardTimeSeries that is a list of scalars. E.g.
        accuracy of a model over epochs/time.
      TENSOR: Used for TensorboardTimeSeries that is a list of tensors. E.g.
        histograms of weights of layer in a model over epoch/time.
      BLOB_SEQUENCE: Used for TensorboardTimeSeries that is a list of blob
        sequences. E.g. set of sample images with labels over epochs/time.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        SCALAR = 1
        TENSOR = 2
        BLOB_SEQUENCE = 3
    tensorboardTimeSeriesId = _messages.StringField(1)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 2)
    values = _messages.MessageField('GoogleCloudAiplatformV1beta1TimeSeriesDataPoint', 3, repeated=True)