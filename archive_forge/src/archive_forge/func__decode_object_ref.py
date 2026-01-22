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
def _decode_object_ref(self, obj: ObjectRefRep) -> Model:
    id = obj['id']
    instance = self._references.get(id)
    if instance is not None:
        warn(f"reference already known '{id}'", BokehUserWarning)
        return instance
    name = obj['name']
    attributes = obj.get('attributes')
    cls = self._resolve_type(name)
    instance = cls.__new__(cls, id=id)
    if instance is None:
        self.error(f"can't instantiate {name}(id={id})")
    self._references[instance.id] = instance
    if not instance._initialized:
        from .has_props import HasProps
        HasProps.__init__(instance)
    if attributes is not None:
        decoded_attributes = {key: self._decode(val) for key, val in attributes.items()}
        for key, val in decoded_attributes.items():
            instance.set_from_json(key, val, setter=self._setter)
    return instance