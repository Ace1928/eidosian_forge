import builtins
import collections
import math
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import convert_to_tensor
def with_combined_batch_dimensions(a, b, output_shape, fn_3d):
    a_sparse = isinstance(a, tf.SparseTensor)
    b_sparse = isinstance(b, tf.SparseTensor)
    batch_shape = b.shape[:-2] if b_sparse else a.shape[:-2]
    batch_size = math.prod(batch_shape)
    a3d_shape = [batch_size] + a.shape[-2:]
    a_3d = tf.sparse.reshape(a, a3d_shape) if a_sparse else tf.reshape(a, a3d_shape)
    b3d_shape = [batch_size] + b.shape[-2:]
    b_3d = tf.sparse.reshape(b, b3d_shape) if b_sparse else tf.reshape(b, b3d_shape)
    result_3d = fn_3d(a_3d, b_3d)
    return tf.sparse.reshape(result_3d, output_shape) if isinstance(result_3d, tf.SparseTensor) else tf.reshape(result_3d, output_shape)