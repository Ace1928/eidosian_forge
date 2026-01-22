from typing import List
from typing import Optional
from typing import Union
from selenium.webdriver.remote.command import Command
from . import interaction
from .key_actions import KeyActions
from .key_input import KeyInput
from .pointer_actions import PointerActions
from .pointer_input import PointerInput
from .wheel_actions import WheelActions
from .wheel_input import WheelInput
def add_key_input(self, name: str) -> KeyInput:
    new_input = KeyInput(name)
    self._add_input(new_input)
    return new_input