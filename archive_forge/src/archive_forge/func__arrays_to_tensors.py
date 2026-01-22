import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union
import numpy as np
import tensorflow as tf
from ray.air._internal.tensorflow_utils import convert_ndarray_batch_to_tf_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
def _arrays_to_tensors(self, numpy_arrays: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[tf.dtypes.DType, Dict[str, tf.dtypes.DType]]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    return convert_ndarray_batch_to_tf_tensor_batch(numpy_arrays, dtypes=dtype)