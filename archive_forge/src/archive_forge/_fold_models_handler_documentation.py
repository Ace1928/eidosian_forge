from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel

        Run all processes to gain stats. It applies algorithms to fold files that gains from learning. It keeps
        stats inside models and models are stored in DataFrame. Columns are matched to the different algos and rows to
        the folds.

        Args:
            :param splitter: Splitter entity.
            :param fold_size: The size of fold.
            :param folds_count: Count of golds.
            :param fold_offset: The offset (count of folds that we want to skip).
            :return: return dict: keys metric to CaseEvaluationResult

        