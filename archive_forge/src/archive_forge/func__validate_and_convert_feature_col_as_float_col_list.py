import base64
import json
import logging
import os
from collections import namedtuple
from typing import (
import numpy as np
import pandas as pd
from pyspark import RDD, SparkContext, cloudpickle
from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
from pyspark.ml.util import (
from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, countDistinct, pandas_udf, rand, struct
from pyspark.sql.types import (
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module
import xgboost
from xgboost import XGBClassifier
from xgboost.compat import is_cudf_available, is_cupy_available
from xgboost.core import Booster, _check_distributed_params
from xgboost.sklearn import DEFAULT_N_ESTIMATORS, XGBModel, _can_use_qdm
from xgboost.training import train as worker_train
from .._typing import ArrayLike
from .data import (
from .params import (
from .utils import (
def _validate_and_convert_feature_col_as_float_col_list(dataset: DataFrame, features_col_names: List[str]) -> List[Column]:
    """Values in feature columns must be integral types or float/double types"""
    feature_cols = []
    for c in features_col_names:
        if isinstance(dataset.schema[c].dataType, DoubleType):
            feature_cols.append(col(c).cast(FloatType()).alias(c))
        elif isinstance(dataset.schema[c].dataType, (FloatType, IntegralType)):
            feature_cols.append(col(c))
        else:
            raise ValueError('Values in feature columns must be integral types or float/double types.')
    return feature_cols