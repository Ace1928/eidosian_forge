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
def _get_distributed_train_params(self, dataset: DataFrame) -> Dict[str, Any]:
    """
        This just gets the configuration params for distributed xgboost
        """
    params = self._gen_xgb_params_dict()
    fit_params = self._gen_fit_params_dict()
    verbose_eval = fit_params.pop('verbose', None)
    params.update(fit_params)
    params['verbose_eval'] = verbose_eval
    classification = self._xgb_cls() == XGBClassifier
    if classification:
        num_classes = int(dataset.select(countDistinct(alias.label)).collect()[0][0])
        if num_classes <= 2:
            params['objective'] = 'binary:logistic'
        else:
            params['objective'] = 'multi:softprob'
            params['num_class'] = num_classes
    else:
        params['objective'] = self.getOrDefault('objective')
    params['num_boost_round'] = self.getOrDefault('n_estimators')
    return params