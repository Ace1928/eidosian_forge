from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Sequence, Union
import numpy as np
def _defineprop(name: str, dtype: type=float, shape: Union[ShapeSpec, Sequence[ShapeSpec]]=tuple()) -> Property:
    """Create, register, and return a property."""
    if isinstance(shape, (int, str)):
        shape = (shape,)
    shape = tuple(shape)
    prop: Property
    if len(shape) == 0:
        prop = ScalarProperty(name, dtype)
    else:
        prop = ArrayProperty(name, dtype, shape)
    all_outputs[name] = prop
    return prop