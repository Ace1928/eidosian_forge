from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TensorboardTimeSeries(_messages.Message):
    """TensorboardTimeSeries maps to times series produced in training runs

  Enums:
    ValueTypeValueValuesEnum: Required. Immutable. Type of
      TensorboardTimeSeries value.

  Fields:
    createTime: Output only. Timestamp when this TensorboardTimeSeries was
      created.
    description: Description of this TensorboardTimeSeries.
    displayName: Required. User provided name of this TensorboardTimeSeries.
      This value should be unique among all TensorboardTimeSeries resources
      belonging to the same TensorboardRun resource (parent resource).
    etag: Used to perform a consistent read-modify-write updates. If not set,
      a blind "overwrite" update happens.
    metadata: Output only. Scalar, Tensor, or Blob metadata for this
      TensorboardTimeSeries.
    name: Output only. Name of the TensorboardTimeSeries.
    pluginData: Data of the current plugin, with the size limited to 65KB.
    pluginName: Immutable. Name of the plugin this time series pertain to.
      Such as Scalar, Tensor, Blob
    updateTime: Output only. Timestamp when this TensorboardTimeSeries was
      last updated.
    valueType: Required. Immutable. Type of TensorboardTimeSeries value.
  """

    class ValueTypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. Type of TensorboardTimeSeries value.

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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1TensorboardTimeSeriesMetadata', 5)
    name = _messages.StringField(6)
    pluginData = _messages.BytesField(7)
    pluginName = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 10)