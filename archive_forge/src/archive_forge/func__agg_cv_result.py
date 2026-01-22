import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from . import callback
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _InnerPredictor,
from .compat import SKLEARN_INSTALLED, _LGBMBaseCrossValidator, _LGBMGroupKFold, _LGBMStratifiedKFold
def _agg_cv_result(raw_results: List[List[_LGBM_BoosterEvalMethodResultType]]) -> List[_LGBM_BoosterEvalMethodResultWithStandardDeviationType]:
    """Aggregate cross-validation results."""
    cvmap: Dict[str, List[float]] = OrderedDict()
    metric_type: Dict[str, bool] = {}
    for one_result in raw_results:
        for one_line in one_result:
            key = f'{one_line[0]} {one_line[1]}'
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, float(np.mean(v)), metric_type[k], float(np.std(v))) for k, v in cvmap.items()]