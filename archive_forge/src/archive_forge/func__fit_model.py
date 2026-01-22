from __future__ import print_function
import os
import time
from .. import CatBoostError
from ..eval.log_config import get_eval_logger
from ..eval.utils import make_dirs_if_not_exists
from ..eval.evaluation_result import CaseEvaluationResult
from ._fold_model import FoldModel
@staticmethod
def _fit_model(pool, case, fold_id, model_path):
    from .. import CatBoost
    make_dirs_if_not_exists(FoldModelsHandler.__MODEL_DIR)
    feature_count = pool.num_col()
    if 'ignored_features' in case.get_params():
        ignored_features = case.get_params()['ignored_features']
        if len(ignored_features) and max(ignored_features) >= feature_count:
            raise CatBoostError('Error: input parameter contains feature indices wich are not available in pool: {}\n Check eval_feature set and ignored features options'.format(ignored_features))
    get_eval_logger().debug('Learn model {} on fold #{}'.format(str(case), fold_id))
    cur_time = time.time()
    instance = CatBoost(params=case.get_params())
    instance.fit(pool)
    instance.save_model(fname=model_path)
    get_eval_logger().debug('Operation was done in {} seconds'.format(time.time() - cur_time))
    return FoldModel(case, model_path, fold_id)