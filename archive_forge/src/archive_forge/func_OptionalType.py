import datetime
import math
import typing as t
from wandb.util import (
def OptionalType(dtype: ConvertableToType) -> UnionType:
    """Function that mimics the Type class API for constructing an "Optional Type".

    This is just a Union[wb_type, NoneType].

    Args:
        dtype (Type): type to be optional

    Returns:
        Type: Optional version of the type.
    """
    return UnionType([TypeRegistry.type_from_dtype(dtype), NoneType()])