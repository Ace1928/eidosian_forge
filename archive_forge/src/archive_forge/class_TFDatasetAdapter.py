import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
class TFDatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    def __init__(self, dataset, class_weight=None, distribution=None):
        """Iniitialize the TFDatasetAdapter.

        Args:
            dataset: The input `tf.data.Dataset` instance.
            class_weight: A map where the keys are integer class ids and values
                are the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`.
            distribution: A `keras.distribution.Distribution` instance. Used to
                shard the input dataset into per worker/process dataset
                instance.
        """
        from keras.src.utils.module_utils import tensorflow as tf
        if not isinstance(dataset, (tf.data.Dataset, tf.distribute.DistributedDataset)):
            raise ValueError(f'Expected argument `dataset` to be a tf.data.Dataset. Received: {dataset}')
        if class_weight is not None:
            dataset = dataset.map(make_class_weight_map_fn(class_weight)).prefetch(tf.data.AUTOTUNE)
        if distribution is not None:
            dataset = distribution.distribute_dataset(dataset)
        self._dataset = dataset

    def get_numpy_iterator(self):
        from keras.src.backend.tensorflow.core import convert_to_numpy
        for batch in self._dataset:
            yield tree.map_structure(convert_to_numpy, batch)

    def get_jax_iterator(self):
        import jax.experimental.sparse as jax_sparse
        from keras.src.backend.jax.core import convert_to_tensor
        from keras.src.backend.tensorflow.core import convert_to_numpy
        from keras.src.utils.module_utils import tensorflow as tf

        def convert_to_jax(x):
            if isinstance(x, tf.SparseTensor):
                values = convert_to_numpy(x.values)
                indices = convert_to_numpy(x.indices)
                return jax_sparse.BCOO((values, indices), shape=x.shape)
            return convert_to_tensor(convert_to_numpy(x))
        for batch in self._dataset:
            yield tree.map_structure(convert_to_jax, batch)

    def get_tf_dataset(self):
        return self._dataset

    def get_torch_dataloader(self):
        return data_adapter_utils.get_torch_dataloader(self._dataset)

    @property
    def num_batches(self):
        cardinality = self._dataset.cardinality
        if callable(cardinality):
            cardinality = int(self._dataset.cardinality())
        else:
            cardinality = int(cardinality)
        if cardinality < 0:
            return None
        return cardinality

    @property
    def batch_size(self):
        first_element_spec = tree.flatten(self._dataset.element_spec)[0]
        return first_element_spec.shape[0]

    @property
    def has_partial_batch(self):
        return None

    @property
    def partial_batch_size(self):
        return None