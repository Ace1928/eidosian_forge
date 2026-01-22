import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
def _get_group_from_constructed_dataset(dataset: Dataset) -> Optional[np.ndarray]:
    group = dataset.get_group()
    error_msg = "Estimators in lightgbm.sklearn should only retrieve query groups from a constructed Dataset. If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/microsoft/LightGBM/issues."
    assert group is None or isinstance(group, np.ndarray), error_msg
    return group