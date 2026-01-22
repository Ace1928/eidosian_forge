from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import Required, String
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..model import Model
class ByID(Selector):
    """ Represents a CSS ID selector query. """

    def __init__(self, query: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(query=query, **kwargs)
    query = Required(String, help='\n    Element CSS ID without ``#`` prefix. Alternatively use ``ByCSS("#id")``.\n    ')