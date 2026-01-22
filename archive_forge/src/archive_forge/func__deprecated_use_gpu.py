import warnings
from typing import Any, List, Optional, Type, Union
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
from .utils import get_class_name
def _deprecated_use_gpu() -> None:
    warnings.warn('`use_gpu` is deprecated since 2.0.0, use `device` instead', FutureWarning)