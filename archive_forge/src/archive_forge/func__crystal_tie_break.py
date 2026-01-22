from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
def _crystal_tie_break(a, b, logs):
    """Given two tuples of prime powers, break ties."""
    return logs.dot(np.abs(a)) < logs.dot(np.abs(b))