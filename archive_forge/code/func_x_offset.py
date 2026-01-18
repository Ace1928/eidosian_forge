from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .input_device import InputDevice
@property
def x_offset(self) -> int:
    return self._x_offset