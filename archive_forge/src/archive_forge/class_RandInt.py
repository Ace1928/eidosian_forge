import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class RandInt(RandBase):
    """Uniform distributed random integer values.
    Please read |SpaceTutorial|.

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, ``low`` must be ``>=1`` and lower values get higher chance to be sampled
    """

    def __init__(self, low: int, high: int, q: int=1, log: bool=False, include_high: bool=True):
        if include_high:
            assert_or_throw(high >= low, ValueError(f'{high} < {low}'))
        else:
            assert_or_throw(high > low, ValueError(f'{high} <= {low}'))
        assert_or_throw(q > 0, ValueError(q))
        if log:
            assert_or_throw(low >= 1.0, ValueError(f'for log sampling, low ({low}) must be greater or equal to 1.0'))
        self.low = low
        self.high = high
        self.include_high = include_high
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_='randint', low=self.low, high=self.high, q=self.q, log=self.log, include_high=self.include_high)

    def generate(self, seed: Any=None) -> float:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.uniform()
        return int(uniform_to_integers(value, self.low, self.high, q=int(self.q), log=self.log, include_high=self.include_high))

    def __repr__(self) -> str:
        return f'RandInt(low={self.low}, high={self.high}, q={self.q}, log={self.log}, include_high={self.include_high})'