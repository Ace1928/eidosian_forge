import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class RandBase(StochasticExpression):
    """Base class for continuous random variables.
    Please read |SpaceTutorial|.

    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, lower values get higher chance to be sampled
    """

    def __init__(self, q: Optional[float]=None, log: bool=False):
        if q is not None:
            assert_or_throw(q > 0, f'{q} <= 0')
        self.q = q
        self.log = log