from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
@error(NOT_A_PROPERTY_OF)
def _check_if_an_attribute_is_a_property_of_a_model(self):
    if self.obj.lookup(self.attr, raises=False):
        return None
    else:
        return f'{self.attr} is not a property of {self.obj}'