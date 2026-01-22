from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
def _fit_models(self, learn_files, fold_id_bias):
    """
        Train models for each algorithm and learn dataset(folds). Than return them.

        Args:
            :param learn_files: Entities of FoldStorage for learning models.
            :return: Dictionary of models where the key is case and the value is models on learn folds
        """
    make_dirs_if_not_exists(FoldModelsHandler.__MODEL_DIR)
    models = {}
    for case in self._cases:
        models[case] = list()
    for file_num, learn_file in enumerate(learn_files):
        pool = FoldModelsHandler._create_pool(learn_file, self._thread_count)
        fold_id = fold_id_bias + file_num
        for case in self._cases:
            model_path = os.path.join(FoldModelsHandler.__MODEL_DIR, FoldModelsHandler._create_model_name(case, fold_id))
            get_eval_logger().debug('For model {} on fold #{} path is {}'.format(str(case), fold_id, model_path))
            fold_model = self._fit_model(pool, case, fold_id, model_path)
            get_eval_logger().info('Model {} on fold #{} was fitted'.format(str(case), fold_id))
            models[case].append(fold_model)
    return models