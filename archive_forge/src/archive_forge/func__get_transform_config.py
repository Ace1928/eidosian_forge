import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union
import numpy as np
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import Deprecated
def _get_transform_config(self) -> Dict[str, Any]:
    return {'batch_size': self.batch_size}