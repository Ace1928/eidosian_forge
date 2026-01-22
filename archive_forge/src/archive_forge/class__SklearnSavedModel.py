import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
class _SklearnSavedModel(_PicklingSavedModel['sklearn.base.BaseEstimator']):
    _log_type = 'sklearn-model-file'
    _path_extension = 'pkl'

    @staticmethod
    def _deserialize(dir_or_file_path: str) -> 'sklearn.base.BaseEstimator':
        with open(dir_or_file_path, 'rb') as file:
            model = _get_cloudpickle().load(file)
        return model

    @staticmethod
    def _validate_obj(obj: Any) -> bool:
        dynamic_sklearn = _get_sklearn()
        return cast(bool, dynamic_sklearn.base.is_classifier(obj) or dynamic_sklearn.base.is_outlier_detector(obj) or dynamic_sklearn.base.is_regressor(obj))

    @staticmethod
    def _serialize(model_obj: 'sklearn.base.BaseEstimator', dir_or_file_path: str) -> None:
        dynamic_cloudpickle = _get_cloudpickle()
        with open(dir_or_file_path, 'wb') as file:
            dynamic_cloudpickle.dump(model_obj, file)