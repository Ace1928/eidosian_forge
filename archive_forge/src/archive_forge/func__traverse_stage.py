import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _traverse_stage(stage):
    from pyspark.ml import Pipeline
    yield stage
    if isinstance(stage, Pipeline):
        original_sub_stages = stage.getStages()
        try:
            iter(original_sub_stages)
        except TypeError:
            raise TypeError(f'Pipeline stages should be iterable, but found object {original_sub_stages}')
        for stage in original_sub_stages:
            yield from _traverse_stage(stage)
    else:
        for _, param_value in _get_stage_type_params(stage).items():
            yield from _traverse_stage(param_value)