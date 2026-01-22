import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar
import numpy as np
import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor
from ._wires import wires_to_json
@classmethod
@lru_cache(1)
def _supported_ops_dict(cls) -> Dict[str, Type[Operator]]:
    """Returns a dict mapping ``Operator`` subclass names to the class."""
    return {op.__name__: op for op in cls.consumes_types()}