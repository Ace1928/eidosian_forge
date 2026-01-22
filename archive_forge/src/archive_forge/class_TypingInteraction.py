from . import interaction
from .input_device import InputDevice
from .interaction import Interaction
from .interaction import Pause
class TypingInteraction(Interaction):

    def __init__(self, source, type_, key) -> None:
        super().__init__(source)
        self.type = type_
        self.key = key

    def encode(self) -> dict:
        return {'type': self.type, 'value': self.key}