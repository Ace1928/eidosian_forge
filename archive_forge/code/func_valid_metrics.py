from abc import abstractmethod
from typing import List
import numpy as np
from scipy.sparse import issparse
from ... import get_config
from .._dist_metrics import (
from ._argkmin import (
from ._argkmin_classmode import (
from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64
from ._radius_neighbors import (
from ._radius_neighbors_classmode import (
@classmethod
def valid_metrics(cls) -> List[str]:
    excluded = {'euclidean', 'sqeuclidean'}
    return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)