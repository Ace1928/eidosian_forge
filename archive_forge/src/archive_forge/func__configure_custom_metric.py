import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
def _configure_custom_metric(feval: Optional[Metric], custom_metric: Optional[Metric]) -> Optional[Metric]:
    if feval is not None:
        link = 'https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html'
        warnings.warn(f'`feval` is deprecated, use `custom_metric` instead.  They have different behavior when custom objective is also used.See {link} for details on the `custom_metric`.')
    if feval is not None and custom_metric is not None:
        raise ValueError('Both `feval` and `custom_metric` are supplied.  Use `custom_metric` instead.')
    eval_metric = custom_metric if custom_metric is not None else feval
    return eval_metric