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
def _validate_gpu_params(self) -> None:
    """Validate the gpu parameters and gpu configurations"""
    if use_cuda(self.getOrDefault(self.device)) or self.getOrDefault(self.use_gpu):
        ss = _get_spark_session()
        sc = ss.sparkContext
        if _is_local(sc):
            get_logger(self.__class__.__name__).warning('You have enabled GPU in spark local mode. Please make sure your local node has at least %d GPUs', self.getOrDefault(self.num_workers))
        else:
            executor_gpus = sc.getConf().get('spark.executor.resource.gpu.amount')
            if executor_gpus is None:
                raise ValueError('The `spark.executor.resource.gpu.amount` is required for training on GPU.')
            if not (ss.version >= '3.4.0' and _is_standalone_or_localcluster(sc)):
                gpu_per_task = sc.getConf().get('spark.task.resource.gpu.amount')
                if gpu_per_task is not None:
                    if float(gpu_per_task) < 1.0:
                        raise ValueError("XGBoost doesn't support GPU fractional configurations. Please set `spark.task.resource.gpu.amount=spark.executor.resource.gpu.amount`")
                    if float(gpu_per_task) > 1.0:
                        get_logger(self.__class__.__name__).warning('%s GPUs for each Spark task is configured, but each XGBoost training task uses only 1 GPU.', gpu_per_task)
                else:
                    raise ValueError('The `spark.task.resource.gpu.amount` is required for training on GPU.')