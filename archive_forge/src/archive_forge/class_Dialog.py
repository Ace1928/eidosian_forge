from __future__ import annotations
import logging # isort:skip
from ...core.enums import Movable, Resizable
from ...core.properties import (
from ..dom import DOMNode
from ..nodes import Node
from .ui_element import UIElement
class Dialog(UIElement):
    """ A floating, movable and resizable container for UI elements.

    .. note::
        This model and all its properties is experimental and may change at any point.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    title = Nullable(Either(String, Instance(DOMNode), Instance(UIElement)), help='\n    The title of the dialog.\n\n    This can be either a plain text string, a DOM node, a UI element or a layout.\n    ')
    content = Required(Either(String, Instance(DOMNode), Instance(UIElement)), help='\n    The contents of this dialog.\n\n    This can be either a plain text string, a DOM node, a UI element or a layout.\n    ')
    pinnable = Bool(default=True, help='\n    Determines whether to allow to pin the dialog.\n\n    A pinned dialog always stays on top of other dialogs. Pinning one dialog\n    unpins any other dialogs.\n    ')
    collapsible = Bool(default=True, help='\n    Determines whether to allow to collapse the dialog.\n\n    A collapsed dialog only shows its title, while its content is hidden from\n    the view. This allows keep a dialog open while having a better accesses\n    to UIs below it.\n\n    .. note::\n        A dialog can be collapsed by scrolling on its title.\n    ')
    minimizable = Bool(default=True, help='\n    Determines whether to allow to minimize the dialog.\n\n    Minimizing a dialog means collapsing it and moving it to a designated\n    "minimization" area in the bottom left corner of the viewport.\n    ')
    maximizable = Bool(default=True, help='\n    Determines whether to allow to maximize the dialog.\n\n    A maximized dialog covers the entire viewport area. Multiple dialogs\n    can be maximized at the same time, but only one will be at the top\n    of the viewport.\n    ')
    closable = Bool(default=True, help="\n    Determines whether to allow to close the dialog.\n\n    Property ``close_action`` determines what happens when a dialog is\n    closed. Note that even if dialog can't be closed through the UI,\n    it can be closed programmatically.\n    ")
    close_action = Enum('hide', 'destroy', default='destroy', help='\n    Determines the action when closing a dialog.\n\n    Options are:\n    * ``"hide"`` - Removes the dialog from the DOM, but keeps its\n        view "alive", so that it can be opened another time.\n    * ``"destroy"`` - Destroys the associated view and the state\n        it stores. A dialog needs to be rebuilt with a fresh state\n        before it can be opened again.\n    ')
    resizable = Enum(Resizable, default='all', help='\n    Determines whether or in which directions a dialog can be resized.\n    ')
    movable = Enum(Movable, default='both', help='\n    Determines whether or in which directions a dialog can be moved.\n    ')
    symmetric = Bool(default=False, help='\n    Determines if resizing one edge or corner affects the opposite one.\n    ')
    top_limit = Nullable(Instance(Node), default=None, help='\n    Optional top movement or resize limit.\n\n    Together with ``bottom_limit``, ``left_limit`` and ``right_limit`` it\n    forms a bounding box for movement and resizing of this dialog.\n    ')
    bottom_limit = Nullable(Instance(Node), default=None, help='\n    Optional bottom movement or resize limit.\n\n    Together with ``top_limit``, ``left_limit`` and ``right_limit`` it\n    forms a bounding box for movement and resizing of this dialog.\n    ')
    left_limit = Nullable(Instance(Node), default=None, help='\n    Optional left movement or resize limit.\n\n    Together with ``top_limit``, ``bottom_limit`` and ``right_limit`` it\n    forms a bounding box for movement and resizing of this dialog.\n    ')
    right_limit = Nullable(Instance(Node), default=None, help='\n    Optional right movement or resize limit.\n\n    Together with ``top_limit``, ``bottom_limit`` and ``left_limit`` it\n    forms a bounding box for movement and resizing of this dialog.\n    ')