from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import NOT_A_PROPERTY_OF
from ..model import Model, Qualified
from .css import Styles
from .ui.ui_element import UIElement
class ValueOf(Placeholder):
    """ A placeholder for the value of a model's property. """

    def __init__(self, obj: Init[HasProps]=Intrinsic, attr: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(obj=obj, attr=attr, **kwargs)
    obj: HasProps = Required(Instance(HasProps), help='\n    The object whose property will be observed.\n    ')
    attr: str = Required(String, help='\n    The name of the property whose value will be observed.\n    ')

    @error(NOT_A_PROPERTY_OF)
    def _check_if_an_attribute_is_a_property_of_a_model(self):
        if self.obj.lookup(self.attr, raises=False):
            return None
        else:
            return f'{self.attr} is not a property of {self.obj}'