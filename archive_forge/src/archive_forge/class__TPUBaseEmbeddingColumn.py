import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
class _TPUBaseEmbeddingColumn(object):
    """Base class for TPU Embedding Column."""

    def __init__(self, categorical_column, max_sequence_length=0, learning_rate_fn=None):
        self._tpu_categorical_column = categorical_column
        self._max_sequence_length = max_sequence_length
        self._learning_rate_fn = learning_rate_fn
        if self.is_sequence_column() and max_sequence_length < 1:
            raise ValueError('max_sequence_length must be greater than 0 for sequence columns. Got max_sequence_length={} for sequence column {}.'.format(max_sequence_length, categorical_column.name))
        if not self.is_sequence_column() and max_sequence_length != 0:
            raise ValueError('Non zero max_seq_length={} specified for non sequence column {}.'.format(max_sequence_length, categorical_column.name))

    def get_combiner(self):
        """Returns the embedding combiner."""
        raise NotImplementedError('not implemented')

    def get_embedding_table_size(self):
        """Returns the embedding table size, tuple of vocab size and dimension."""
        raise NotImplementedError('not implemented')

    def get_feature_key_name(self):
        """Returns the feature key name in the features dict."""
        raise NotImplementedError('not impl')

    def get_weight_key_name(self):
        """Return the key name for weights."""
        raise NotImplementedError('not impl')

    def get_embedding_var_name(self):
        """Returns the embedding variable name.

    Feature key name and embedding variable name are usually one-to-one mapping.
    But for shared embedding columns, it is many-to-one mapping.
    """
        raise NotImplementedError('not impl')

    def get_initializer(self):
        """Returns the initializer."""
        raise NotImplementedError('not impl')

    def is_categorical_column_weighted(self):
        """Check if the categorical column of the embedding column is weighted."""
        raise NotImplementedError('not impl')

    def is_sequence_column(self):
        return isinstance(self._tpu_categorical_column, _SUPPORTED_SEQUENCE_COLUMNS)

    def get_max_sequence_length(self):
        return self._max_sequence_length

    def get_learning_rate_fn(self):
        return self._learning_rate_fn

    def get_sequence_length_feature_key_name(self):
        """Get the key for the associated sequence length feature."""
        return get_sequence_length_feature_key_name_from_feature_key_name(self.get_feature_key_name())