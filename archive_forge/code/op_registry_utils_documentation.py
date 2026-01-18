import functools
from inspect import signature
from .common_op_utils import _basic_validation

    Decorator function to register the given ``op`` in the provided
    ``op_table``
    