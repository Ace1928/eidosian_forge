from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
@abstract
class BaseText(Model):
    """
    Base class for renderers of text content of various kinds.
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and 'text' not in kwargs:
            kwargs['text'] = args[0]
        super().__init__(**kwargs)
    text = Required(String, help='\n    The text value to render.\n    ')