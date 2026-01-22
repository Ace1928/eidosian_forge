from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
class _IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that drops erroneous elements from its input."""

    def __init__(self, input_dataset, log_warning, name=None):
        """See `Dataset.ignore_errors` for details."""
        self._input_dataset = input_dataset
        self._name = name
        variant_tensor = gen_experimental_dataset_ops.ignore_errors_dataset(self._input_dataset._variant_tensor, log_warning=log_warning, **self._flat_structure)
        super().__init__(input_dataset, variant_tensor)