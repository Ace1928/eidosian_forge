import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def _fill_funcs(self, obj: Dict[str, Any]) -> Dict[str, Any]:

    def realize_func(path: List[Any]) -> None:
        r: Any = obj
        for p in path[:-1]:
            r = r[p]
        r[path[-1]] = r[path[-1]]()
    for path in self._func_positions:
        realize_func(path)
    return obj