from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from .ui_element import UIElement
@abstract
class Icon(UIElement):
    """ An abstract base class for icon elements.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    size = Either(Int, FontSize, default='1em', help='\n    The size of the icon. This can be either a number of pixels, or a CSS\n    length string (see https://developer.mozilla.org/en-US/docs/Web/CSS/length).\n    ')