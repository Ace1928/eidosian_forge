from __future__ import annotations
import logging # isort:skip
import base64
import datetime as dt
import sys
from array import array as TypedArray
from math import isinf, isnan
from types import SimpleNamespace
from typing import (
import numpy as np
from ..util.dataclasses import (
from ..util.dependencies import uses_pandas
from ..util.serialization import (
from ..util.warnings import BokehUserWarning, warn
from .types import ID
def _resolve_type(self, type: str) -> type[Model]:
    from ..model import Model
    cls = Model.model_class_reverse_map.get(type)
    if cls is not None:
        if issubclass(cls, Model):
            return cls
        else:
            self.error(f"object of type '{type}' is not a subclass of 'Model'")
    elif type == 'Figure':
        from ..plotting import figure
        return figure
    else:
        self.error(f"can't resolve type '{type}'")