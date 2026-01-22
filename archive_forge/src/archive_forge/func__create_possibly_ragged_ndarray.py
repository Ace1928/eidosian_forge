from typing import Any, Sequence, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.util import PublicAPI
def _create_possibly_ragged_ndarray(values: Union[np.ndarray, 'ABCSeries', Sequence[Any]]) -> np.ndarray:
    """
    Create a possibly ragged ndarray.
    Using the np.array() constructor will fail to construct a ragged ndarray that has a
    uniform first dimension (e.g. uniform channel dimension in imagery). This function
    catches this failure and tries a create-and-fill method to construct the ragged
    ndarray.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)
            return np.array(values, copy=False)
    except ValueError as e:
        error_str = str(e)
        if 'could not broadcast input array from shape' in error_str or 'The requested array has an inhomogeneous shape' in error_str:
            return create_ragged_ndarray(values)
        else:
            raise e from None