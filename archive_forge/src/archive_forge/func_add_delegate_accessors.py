from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
def add_delegate_accessors(cls):
    cls._add_delegate_accessors(delegate, accessors, typ, overwrite=overwrite, accessor_mapping=accessor_mapping, raise_on_missing=raise_on_missing)
    return cls