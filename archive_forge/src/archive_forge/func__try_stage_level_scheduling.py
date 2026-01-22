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
def _try_stage_level_scheduling(self, rdd: RDD) -> RDD:
    """Try to enable stage-level scheduling"""
    if self._skip_stage_level_scheduling():
        return rdd
    ss = _get_spark_session()
    executor_cores = ss.sparkContext.getConf().get('spark.executor.cores')
    assert executor_cores is not None
    spark_plugins = ss.conf.get('spark.plugins', ' ')
    assert spark_plugins is not None
    spark_rapids_sql_enabled = ss.conf.get('spark.rapids.sql.enabled', 'true')
    assert spark_rapids_sql_enabled is not None
    task_cores = int(executor_cores) if 'com.nvidia.spark.SQLPlugin' in spark_plugins and 'true' == spark_rapids_sql_enabled.lower() else int(executor_cores) // 2 + 1
    task_gpus = 1.0
    treqs = TaskResourceRequests().cpus(task_cores).resource('gpu', task_gpus)
    rp = ResourceProfileBuilder().require(treqs).build
    self.logger.info('XGBoost training tasks require the resource(cores=%s, gpu=%s).', task_cores, task_gpus)
    return rdd.withResources(rp)