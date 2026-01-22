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
def _get_xgb_parameters(self, dataset: DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    train_params = self._get_distributed_train_params(dataset)
    booster_params, train_call_kwargs_params = self._get_xgb_train_call_args(train_params)
    cpu_per_task = int(_get_spark_session().sparkContext.getConf().get('spark.task.cpus', '1'))
    dmatrix_kwargs = {'nthread': cpu_per_task, 'feature_types': self.getOrDefault('feature_types'), 'feature_names': self.getOrDefault('feature_names'), 'feature_weights': self.getOrDefault('feature_weights'), 'missing': float(self.getOrDefault('missing'))}
    if dmatrix_kwargs['feature_types'] is not None:
        dmatrix_kwargs['enable_categorical'] = True
    booster_params['nthread'] = cpu_per_task
    booster_params = {k: v for k, v in booster_params.items() if v is not None}
    train_call_kwargs_params = {k: v for k, v in train_call_kwargs_params.items() if v is not None}
    dmatrix_kwargs = {k: v for k, v in dmatrix_kwargs.items() if v is not None}
    return (booster_params, train_call_kwargs_params, dmatrix_kwargs)