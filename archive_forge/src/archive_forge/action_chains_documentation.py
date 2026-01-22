from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from .actions.action_builder import ActionBuilder
from .actions.key_input import KeyInput
from .actions.pointer_input import PointerInput
from .actions.wheel_input import ScrollOrigin
from .actions.wheel_input import WheelInput
from .utils import keys_to_typing
Scrolls by provided amount based on a provided origin. The scroll
        origin is either the center of an element or the upper left of the
        viewport plus any offsets. If the origin is an element, and the element
        is not in the viewport, the bottom of the element will first be
        scrolled to the bottom of the viewport.

        :Args:
         - origin: Where scroll originates (viewport or element center) plus provided offsets.
         - delta_x: Distance along X axis to scroll using the wheel. A negative value scrolls left.
         - delta_y: Distance along Y axis to scroll using the wheel. A negative value scrolls up.

         :Raises: If the origin with offset is outside the viewport.
          - MoveTargetOutOfBoundsException - If the origin with offset is outside the viewport.
        