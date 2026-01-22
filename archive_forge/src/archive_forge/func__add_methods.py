from __future__ import annotations
import operator
from pandas.core.dtypes.generic import (
from pandas.core.ops import roperator
def _add_methods(cls, new_methods) -> None:
    for name, method in new_methods.items():
        setattr(cls, name, method)