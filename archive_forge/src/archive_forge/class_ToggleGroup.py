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
class ToggleGroup(Action):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    groups = List(Instance('.models.renderers.RendererGroup'))