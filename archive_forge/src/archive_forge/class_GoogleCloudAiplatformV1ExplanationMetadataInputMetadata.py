from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationMetadataInputMetadata(_messages.Message):
    """Metadata of the input of a feature. Fields other than
  InputMetadata.input_baselines are applicable only for Models that are using
  Vertex AI-provided images for Tensorflow.

  Enums:
    EncodingValueValuesEnum: Defines how the feature is encoded into the input
      tensor. Defaults to IDENTITY.

  Fields:
    denseShapeTensorName: Specifies the shape of the values of the input if
      the input is a sparse representation. Refer to Tensorflow documentation
      for more details:
      https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor.
    encodedBaselines: A list of baselines for the encoded tensor. The shape of
      each baseline should match the shape of the encoded tensor. If a scalar
      is provided, Vertex AI broadcasts to the same shape as the encoded
      tensor.
    encodedTensorName: Encoded tensor is a transformation of the input tensor.
      Must be provided if choosing Integrated Gradients attribution or XRAI
      attribution and the input tensor is not differentiable. An encoded
      tensor is generated if the input tensor is encoded by a lookup table.
    encoding: Defines how the feature is encoded into the input tensor.
      Defaults to IDENTITY.
    featureValueDomain: The domain details of the input feature value. Like
      min/max, original mean or standard deviation if normalized.
    groupName: Name of the group that the input belongs to. Features with the
      same group name will be treated as one feature when computing
      attributions. Features grouped together can have different shapes in
      value. If provided, there will be one single attribution generated in
      Attribution.feature_attributions, keyed by the group name.
    indexFeatureMapping: A list of feature names for each index in the input
      tensor. Required when the input InputMetadata.encoding is
      BAG_OF_FEATURES, BAG_OF_FEATURES_SPARSE, INDICATOR.
    indicesTensorName: Specifies the index of the values of the input tensor.
      Required when the input tensor is a sparse representation. Refer to
      Tensorflow documentation for more details:
      https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor.
    inputBaselines: Baseline inputs for this feature. If no baseline is
      specified, Vertex AI chooses the baseline for this feature. If multiple
      baselines are specified, Vertex AI returns the average attributions
      across them in Attribution.feature_attributions. For Vertex AI-provided
      Tensorflow images (both 1.x and 2.x), the shape of each baseline must
      match the shape of the input tensor. If a scalar is provided, we
      broadcast to the same shape as the input tensor. For custom images, the
      element of the baselines must be in the same format as the feature's
      input in the instance[]. The schema of any single instance may be
      specified via Endpoint's DeployedModels' Model's PredictSchemata's
      instance_schema_uri.
    inputTensorName: Name of the input tensor for this feature. Required and
      is only applicable to Vertex AI-provided images for Tensorflow.
    modality: Modality of the feature. Valid values are: numeric, image.
      Defaults to numeric.
    visualization: Visualization configurations for image explanation.
  """

    class EncodingValueValuesEnum(_messages.Enum):
        """Defines how the feature is encoded into the input tensor. Defaults to
    IDENTITY.

    Values:
      ENCODING_UNSPECIFIED: Default value. This is the same as IDENTITY.
      IDENTITY: The tensor represents one feature.
      BAG_OF_FEATURES: The tensor represents a bag of features where each
        index maps to a feature. InputMetadata.index_feature_mapping must be
        provided for this encoding. For example: ``` input = [27, 6.0, 150]
        index_feature_mapping = ["age", "height", "weight"] ```
      BAG_OF_FEATURES_SPARSE: The tensor represents a bag of features where
        each index maps to a feature. Zero values in the tensor indicates
        feature being non-existent. InputMetadata.index_feature_mapping must
        be provided for this encoding. For example: ``` input = [2, 0, 5, 0,
        1] index_feature_mapping = ["a", "b", "c", "d", "e"] ```
      INDICATOR: The tensor is a list of binaries representing whether a
        feature exists or not (1 indicates existence).
        InputMetadata.index_feature_mapping must be provided for this
        encoding. For example: ``` input = [1, 0, 1, 0, 1]
        index_feature_mapping = ["a", "b", "c", "d", "e"] ```
      COMBINED_EMBEDDING: The tensor is encoded into a 1-dimensional array
        represented by an encoded tensor. InputMetadata.encoded_tensor_name
        must be provided for this encoding. For example: ``` input = ["This",
        "is", "a", "test", "."] encoded = [0.1, 0.2, 0.3, 0.4, 0.5] ```
      CONCAT_EMBEDDING: Select this encoding when the input tensor is encoded
        into a 2-dimensional array represented by an encoded tensor.
        InputMetadata.encoded_tensor_name must be provided for this encoding.
        The first dimension of the encoded tensor's shape is the same as the
        input tensor's shape. For example: ``` input = ["This", "is", "a",
        "test", "."] encoded = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.1, 0.4,
        0.3, 0.5], [0.5, 0.1, 0.3, 0.5, 0.4], [0.5, 0.3, 0.1, 0.2, 0.4], [0.4,
        0.3, 0.2, 0.5, 0.1]] ```
    """
        ENCODING_UNSPECIFIED = 0
        IDENTITY = 1
        BAG_OF_FEATURES = 2
        BAG_OF_FEATURES_SPARSE = 3
        INDICATOR = 4
        COMBINED_EMBEDDING = 5
        CONCAT_EMBEDDING = 6
    denseShapeTensorName = _messages.StringField(1)
    encodedBaselines = _messages.MessageField('extra_types.JsonValue', 2, repeated=True)
    encodedTensorName = _messages.StringField(3)
    encoding = _messages.EnumField('EncodingValueValuesEnum', 4)
    featureValueDomain = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataInputMetadataFeatureValueDomain', 5)
    groupName = _messages.StringField(6)
    indexFeatureMapping = _messages.StringField(7, repeated=True)
    indicesTensorName = _messages.StringField(8)
    inputBaselines = _messages.MessageField('extra_types.JsonValue', 9, repeated=True)
    inputTensorName = _messages.StringField(10)
    modality = _messages.StringField(11)
    visualization = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataInputMetadataVisualization', 12)