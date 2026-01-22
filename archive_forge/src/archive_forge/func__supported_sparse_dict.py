from functools import lru_cache
from typing import Dict, Generic, Tuple, Type, TypeVar, Union, cast
import numpy as np
from scipy.sparse import (
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
@classmethod
@lru_cache(1)
def _supported_sparse_dict(cls) -> Dict[str, Type[Union[SparseArray, SparseMatrix]]]:
    """Returns a dict mapping sparse array class names to the class."""
    return {op.__name__: op for op in cls.consumes_types()}