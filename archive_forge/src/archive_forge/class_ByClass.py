from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Required, String
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..model import Model
class ByClass(Selector):
    """ Represents a CSS class selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    CSS class name without ``.`` prefix. Alternatively use ``ByCSS(".class")``.\n    ')