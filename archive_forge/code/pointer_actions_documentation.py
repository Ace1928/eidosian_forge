from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput

        Args:
        - source: PointerInput instance
        - duration: override the default 250 msecs of DEFAULT_MOVE_DURATION in source
        