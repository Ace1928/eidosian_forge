from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
@staticmethod
def _remove_model_dir():
    try:
        if os.path.exists(FoldModelsHandler.__MODEL_DIR):
            os.rmdir(FoldModelsHandler.__MODEL_DIR)
    except OSError as err:
        get_eval_logger().warning(str(err))