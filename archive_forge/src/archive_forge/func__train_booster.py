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
def _train_booster(pandas_df_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Takes in an RDD partition and outputs a booster for that partition after
            going through the Rabit Ring protocol

            """
    from pyspark import BarrierTaskContext
    context = BarrierTaskContext.get()
    dev_ordinal = None
    use_qdm = _can_use_qdm(booster_params.get('tree_method', None))
    if run_on_gpu:
        dev_ordinal = context.partitionId() if is_local else _get_gpu_id(context)
        booster_params['device'] = 'cuda:' + str(dev_ordinal)
        use_qdm = use_qdm and is_cudf_available()
        get_logger('XGBoost-PySpark').info('Leveraging %s to train with QDM: %s', booster_params['device'], 'on' if use_qdm else 'off')
    if use_qdm and booster_params.get('max_bin', None) is not None:
        dmatrix_kwargs['max_bin'] = booster_params['max_bin']
    _rabit_args = {}
    if context.partitionId() == 0:
        _rabit_args = _get_rabit_args(context, num_workers)
    worker_message = {'rabit_msg': _rabit_args, 'use_qdm': use_qdm}
    messages = context.allGather(message=json.dumps(worker_message))
    if len(set((json.loads(x)['use_qdm'] for x in messages))) != 1:
        raise RuntimeError("The workers' cudf environments are in-consistent ")
    _rabit_args = json.loads(messages[0])['rabit_msg']
    evals_result: Dict[str, Any] = {}
    with CommunicatorContext(context, **_rabit_args):
        dtrain, dvalid = create_dmatrix_from_partitions(pandas_df_iter, feature_prop.features_cols_names, dev_ordinal, use_qdm, dmatrix_kwargs, enable_sparse_data_optim=feature_prop.enable_sparse_data_optim, has_validation_col=feature_prop.has_validation_col)
        if dvalid is not None:
            dval = [(dtrain, 'training'), (dvalid, 'validation')]
        else:
            dval = None
        booster = worker_train(params=booster_params, dtrain=dtrain, evals=dval, evals_result=evals_result, **train_call_kwargs_params)
    context.barrier()
    if context.partitionId() == 0:
        yield pd.DataFrame(data={'config': [booster.save_config()], 'booster': [booster.save_raw('json').decode('utf-8')]})