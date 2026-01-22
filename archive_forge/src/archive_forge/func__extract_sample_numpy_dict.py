from typing import Dict, Union
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from mlflow.utils.autologging_utils import (
def _extract_sample_numpy_dict(input_numpy_features_dict: Dict[str, np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Extracts `INPUT_EXAMPLE_SAMPLE_ROWS` sample from next_input
    as numpy array of dict(str -> ndarray) type.

    Args:
        input_numpy_features_dict: A tensor or numpy array

    Returns:
        A slice (limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of same type as next_input.
        Returns `None` if the type of `input_numpy_features_dict` is unsupported.

    Examples
    --------
    when next_input is dict:
    >>> input_data = {"a": np.array([1, 2, 3, 4, 5, 6, 7, 8])}
    >>> _extract_sample_numpy_dict(input_data)
    {'a': array([1, 2, 3, 4, 5])}

    """
    sliced_data_as_numpy = None
    if isinstance(input_numpy_features_dict, dict):
        sliced_data_as_numpy = {k: _extract_input_example_from_tensor_or_ndarray(v) for k, v in input_numpy_features_dict.items()}
    return sliced_data_as_numpy