import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
@staticmethod
def _is_interesting(x: Any) -> bool:
    if isinstance(x, (list, tuple)):
        return all((v is None for v in x))
    return x is None or x == {}