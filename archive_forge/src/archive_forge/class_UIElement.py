from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..css import Styles, StyleSheet
from ..nodes import Node
@abstract
class UIElement(Model):
    """ Base class for user interface elements.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    visible = Bool(default=True, help='\n    Whether the component should be displayed on screen.\n    ')
    css_classes = List(String, default=[], help='\n    A list of additional CSS classes to add to the underlying DOM element.\n    ').accepts(Seq(String), lambda x: list(x))
    css_variables = Dict(String, Instance(Node), default={}, help="\n    Allows to define dynamically computed CSS variables.\n\n    This can be used, for example, to coordinate positioning and styling\n    between canvas' renderers and/or visuals and HTML-based UI elements.\n\n    Variables defined here are equivalent to setting the same variables\n    under ``:host { ... }`` in a CSS stylesheet.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ")
    styles = Either(Dict(String, Nullable(String)), Instance(Styles), default={}, help='\n    Inline CSS styles applied to the underlying DOM element.\n    ')
    stylesheets = List(Either(Instance(StyleSheet), String, Dict(String, Either(Dict(String, Nullable(String)), Instance(Styles)))), help="\n    Additional style-sheets to use for the underlying DOM element.\n\n    Note that all bokeh's components use shadow DOM, thus any included style\n    sheets must reflect that, e.g. use ``:host`` CSS pseudo selector to access\n    the root DOM element.\n    ")
    context_menu = Nullable(Instance('.models.ui.Menu'), default=None, help='\n    A menu to display when user right clicks on the component.\n\n    .. note::\n        Use shift key when right clicking to display the native context menu.\n    ')