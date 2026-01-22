from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
@DeveloperAPI
class BlockFormat(str, Enum):
    """Internal Dataset block format enum."""
    PANDAS = 'pandas'
    ARROW = 'arrow'
    SIMPLE = 'simple'