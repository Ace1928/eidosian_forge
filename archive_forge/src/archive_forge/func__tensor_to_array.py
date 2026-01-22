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
def _tensor_to_array(self, tensor: tf.Tensor) -> np.ndarray:
    if not isinstance(tensor, tf.Tensor):
        raise ValueError(f'Expected the model to return either a tf.Tensor or a dict of tf.Tensor, but got {type(tensor)} instead. To support models with different output types, subclass TensorflowPredictor and override the `call_model` method to process the output into either torch.Tensor or Dict[str, torch.Tensor].')
    return tensor.numpy()