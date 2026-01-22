from __future__ import annotations
import logging # isort:skip
from ...core.enums import Movable, Resizable
from ...core.properties import (
from ..dom import DOMNode
from ..nodes import Node
from .ui_element import UIElement
 A floating, movable and resizable container for UI elements.

    .. note::
        This model and all its properties is experimental and may change at any point.
    