import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union
import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import Deprecated
def _transform_numpy(self, np_data: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    return self.fn(np_data)