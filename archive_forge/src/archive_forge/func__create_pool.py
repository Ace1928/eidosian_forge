from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
@staticmethod
def _create_pool(fold_file, thread_count=-1):
    from .. import Pool
    data_pool = Pool(fold_file.path(), column_description=fold_file.column_description(), delimiter=fold_file.get_separator(), thread_count=thread_count)
    return data_pool