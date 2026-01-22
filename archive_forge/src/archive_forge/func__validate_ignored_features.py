import json
from .. import CatBoostError
from ..eval.factor_utils import FactorUtils
from ..core import _NumpyAwareEncoder
@staticmethod
def _validate_ignored_features(ignored_features, eval_features):
    for eval_feature in eval_features:
        if eval_feature in ignored_features:
            raise CatBoostError('Feature {} is in ignored set and in tmp-features set at the same time'.format(eval_feature))