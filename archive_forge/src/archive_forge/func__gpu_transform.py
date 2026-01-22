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
def _gpu_transform(self) -> bool:
    """If gpu is used to do the prediction, true to gpu prediction"""
    if _is_local(_get_spark_session().sparkContext):
        return use_cuda(self.getOrDefault(self.device))
    gpu_per_task = _get_spark_session().sparkContext.getConf().get('spark.task.resource.gpu.amount')
    if gpu_per_task is None:
        if use_cuda(self.getOrDefault(self.device)):
            get_logger('XGBoost-PySpark').warning('Do the prediction on the CPUs since no gpu configurations are set')
        return False
    return use_cuda(self.getOrDefault(self.device))