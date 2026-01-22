import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def _determine_direction_for_numeric_split(fval: float, threshold: float, missing_type_str: str, default_left: bool) -> str:
    missing_type = _MissingType(missing_type_str)
    if math.isnan(fval) and missing_type != _MissingType.NAN:
        fval = 0.0
    if missing_type == _MissingType.ZERO and _is_zero(fval) or (missing_type == _MissingType.NAN and math.isnan(fval)):
        direction = 'left' if default_left else 'right'
    else:
        direction = 'left' if fval <= threshold else 'right'
    return direction