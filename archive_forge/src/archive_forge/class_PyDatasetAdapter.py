import multiprocessing.dummy
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
class PyDatasetAdapter(DataAdapter):
    """Adapter for `keras.utils.PyDataset` instances."""

    def __init__(self, x, class_weight=None, shuffle=False):
        self.py_dataset = x
        self.class_weight = class_weight
        self.enqueuer = None
        self.shuffle = shuffle
        self._output_signature = None

    def _set_tf_output_signature(self):
        from keras.src.utils.module_utils import tensorflow as tf

        def get_tensor_spec(x):
            shape = x.shape
            if len(shape) < 1:
                raise ValueError(f'The arrays returned by PyDataset.__getitem__() must be at least rank 1. Received: {x} of rank {len(x.shape)}')
            shape = list(shape)
            shape[0] = None
            dtype = backend.standardize_dtype(x.dtype)
            return tf.TensorSpec(shape=shape, dtype=dtype)
        batch = self.py_dataset[0]
        batch = self._standardize_batch(batch)
        self._output_signature = tree.map_structure(get_tensor_spec, batch)

    def _standardize_batch(self, batch):
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, np.ndarray):
            batch = (batch,)
        if isinstance(batch, list):
            batch = tuple(batch)
        if not isinstance(batch, tuple) or len(batch) not in {1, 2, 3}:
            raise ValueError(f'PyDataset.__getitem__() must return a tuple or a dict. If a tuple, it must be ordered either (input,) or (inputs, targets) or (inputs, targets, sample_weights). Received: {str(batch)[:100]}... of type {type(batch)}')
        if self.class_weight is not None:
            if len(batch) == 3:
                raise ValueError('You cannot specify `class_weight` and `sample_weight` at the same time.')
            if len(batch) == 2:
                sw = data_adapter_utils.class_weight_to_sample_weights(batch[1], self.class_weight)
                batch = batch + (sw,)
        return batch

    def _make_multiprocessed_generator_fn(self):
        workers = self.py_dataset.workers
        use_multiprocessing = self.py_dataset.use_multiprocessing
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                self.enqueuer = OrderedEnqueuer(self.py_dataset, use_multiprocessing=use_multiprocessing, shuffle=self.shuffle)
                self.enqueuer.start(workers=workers, max_queue_size=self.py_dataset.max_queue_size)
                return self.enqueuer.get()
        else:

            def generator_fn():
                order = range(len(self.py_dataset))
                if self.shuffle:
                    order = list(order)
                    random.shuffle(order)
                for i in order:
                    yield self.py_dataset[i]
        return generator_fn

    def _get_iterator(self):
        gen_fn = self._make_multiprocessed_generator_fn()
        for i, batch in enumerate(gen_fn()):
            batch = self._standardize_batch(batch)
            yield batch
            if i >= len(self.py_dataset) - 1 and self.enqueuer:
                self.enqueuer.stop()

    def get_numpy_iterator(self):
        return data_adapter_utils.get_numpy_iterator(self._get_iterator())

    def get_jax_iterator(self):
        return data_adapter_utils.get_jax_iterator(self._get_iterator())

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf
        if self._output_signature is None:
            self._set_tf_output_signature()
        ds = tf.data.Dataset.from_generator(self._get_iterator, output_signature=self._output_signature)
        if self.shuffle:
            ds = ds.shuffle(8)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_torch_dataloader(self):
        return data_adapter_utils.get_torch_dataloader(self._get_iterator())

    def on_epoch_end(self):
        if self.enqueuer:
            self.enqueuer.stop()
        self.py_dataset.on_epoch_end()

    @property
    def num_batches(self):
        return len(self.py_dataset)

    @property
    def batch_size(self):
        return None