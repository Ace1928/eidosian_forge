import collections
import inspect
import warnings
from functools import wraps
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import regularizers
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.distribution import distribution_lib
from keras.src.layers import input_spec
from keras.src.metrics.metric import Metric
from keras.src.ops.operation import Operation
from keras.src.utils import python_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils import tree
def _check_quantize_args(self, mode, compute_dtype):
    if not self.built:
        raise ValueError(f"Cannot quantize a layer that isn't yet built. Layer '{self.name}' (of type '{self.__class__.__name__}') is not built yet.")
    if mode not in ('int8',):
        raise ValueError(f"`quantize` must be one of ('int8'). Received: mode={mode}")
    if mode == 'int8' and compute_dtype == 'float16':
        raise ValueError(f"Quantization mode='{mode}' doesn't work well with compute_dtype='float16'. Consider loading model/layer with another dtype policy such as 'mixed_bfloat16' or 'mixed_float16' before calling `quantize()`.")