from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export
class _AssertCardinalityDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that assert the cardinality of its input."""

    def __init__(self, input_dataset, expected_cardinality):
        self._input_dataset = input_dataset
        self._expected_cardinality = ops.convert_to_tensor(expected_cardinality, dtype=dtypes.int64, name='expected_cardinality')
        variant_tensor = ged_ops.assert_cardinality_dataset(self._input_dataset._variant_tensor, self._expected_cardinality, **self._flat_structure)
        super(_AssertCardinalityDataset, self).__init__(input_dataset, variant_tensor)