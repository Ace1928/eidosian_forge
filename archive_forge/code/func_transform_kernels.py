import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
def transform_kernels(kernels, func, n_gates):
    """Transforms kernel for each gate separately using given function.

    Args:
        kernels: Stacked array of kernels for individual gates.
        func: Function applied to kernel of each gate.
        n_gates: Number of gates (4 for LSTM, 3 for GRU).

    Returns:
        Stacked array of transformed kernels.
    """
    return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])