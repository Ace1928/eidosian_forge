from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple
import numpy as np
import pyarrow
import tensorflow as tf
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
Convert a NumPy ndarray batch to a TensorFlow Tensor batch.

    Args:
        ndarray: A (dict of) NumPy ndarray(s) that we wish to convert to a TensorFlow
            Tensor.
        dtype: A (dict of) TensorFlow dtype(s) for the created tensor; if None, the
            dtype will be inferred from the NumPy ndarray data.

    Returns: A (dict of) TensorFlow Tensor(s).
    