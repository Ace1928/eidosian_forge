import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import MapProto, OptionalProto, SequenceProto, TensorProto, helper, subbyte
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
Create random integer array for backend/test/case/node.

    Args:
        input_shape: The shape for the returned integer array.
        dtype: The NumPy data type for the returned integer array.
        seed: The seed for np.random.

    Returns:
        np.ndarray: Random integer array.
    