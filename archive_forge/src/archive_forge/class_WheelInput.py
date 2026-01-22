from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .input_device import InputDevice
class WheelInput(InputDevice):

    def __init__(self, name) -> None:
        super().__init__(name=name)
        self.name = name
        self.type = interaction.WHEEL

    def encode(self) -> dict:
        return {'type': self.type, 'id': self.name, 'actions': self.actions}

    def create_scroll(self, x: int, y: int, delta_x: int, delta_y: int, duration: int, origin) -> None:
        if isinstance(origin, WebElement):
            origin = {'element-6066-11e4-a52e-4f735466cecf': origin.id}
        self.add_action({'type': 'scroll', 'x': x, 'y': y, 'deltaX': delta_x, 'deltaY': delta_y, 'duration': duration, 'origin': origin})

    def create_pause(self, pause_duration: float) -> None:
        self.add_action({'type': 'pause', 'duration': int(pause_duration * 1000)})