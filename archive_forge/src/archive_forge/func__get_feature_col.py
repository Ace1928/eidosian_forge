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
def _get_feature_col(self, dataset: DataFrame) -> Tuple[List[Column], Optional[List[str]]]:
    """XGBoost model trained with features_cols parameter can also predict
        vector or array feature type. But first we need to check features_cols
        and then featuresCol
        """
    if self.getOrDefault(self.enable_sparse_data_optim):
        feature_col_names = None
        features_col = _get_unwrapped_vec_cols(col(self.getOrDefault(self.featuresCol)))
        return (features_col, feature_col_names)
    feature_col_names = self.getOrDefault(self.features_cols)
    features_col = []
    if feature_col_names and set(feature_col_names).issubset(set(dataset.columns)):
        features_col = _validate_and_convert_feature_col_as_float_col_list(dataset, feature_col_names)
    else:
        feature_col_names = None
        features_col.append(_validate_and_convert_feature_col_as_array_col(dataset, self.getOrDefault(self.featuresCol)))
    return (features_col, feature_col_names)