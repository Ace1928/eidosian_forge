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
def _skip_stage_level_scheduling(self) -> bool:
    """Check if stage-level scheduling is not needed,
        return true to skip stage-level scheduling"""
    if use_cuda(self.getOrDefault(self.device)) or self.getOrDefault(self.use_gpu):
        ss = _get_spark_session()
        sc = ss.sparkContext
        if ss.version < '3.4.0':
            self.logger.info('Stage-level scheduling in xgboost requires spark version 3.4.0+')
            return True
        if not _is_standalone_or_localcluster(sc):
            self.logger.info('Stage-level scheduling in xgboost requires spark standalone or local-cluster mode')
            return True
        executor_cores = sc.getConf().get('spark.executor.cores')
        executor_gpus = sc.getConf().get('spark.executor.resource.gpu.amount')
        if executor_cores is None or executor_gpus is None:
            self.logger.info('Stage-level scheduling in xgboost requires spark.executor.cores, spark.executor.resource.gpu.amount to be set.')
            return True
        if int(executor_cores) == 1:
            self.logger.info('Stage-level scheduling in xgboost requires spark.executor.cores > 1 ')
            return True
        if int(executor_gpus) > 1:
            self.logger.info('Stage-level scheduling in xgboost will not work when spark.executor.resource.gpu.amount>1')
            return True
        task_gpu_amount = sc.getConf().get('spark.task.resource.gpu.amount')
        if task_gpu_amount is None:
            return False
        if float(task_gpu_amount) == float(executor_gpus):
            return True
        return False
    return True