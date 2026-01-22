import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
class TensorflowSparseSliceable(TensorflowSliceable):

    def __init__(self, array):
        super().__init__(to_tensorflow_sparse_wrapper(array))

    @property
    def shape(self):
        return self.array.sparse.shape

    def __getitem__(self, indices):
        return slice_tensorflow_sparse_wrapper(self.array, indices)

    @classmethod
    def convert_to_tf_dataset_compatible(cls, x):
        return to_tensorflow_sparse_wrapper(x)

    @classmethod
    def convert_to_jax_compatible(cls, x):
        return data_adapter_utils.tf_sparse_to_jax_sparse(x)

    @classmethod
    def convert_to_torch_compatible(cls, x):
        from keras.src.utils.module_utils import tensorflow as tf
        return tf.sparse.to_dense(x)