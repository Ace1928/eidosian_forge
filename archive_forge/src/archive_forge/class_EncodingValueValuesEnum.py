from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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