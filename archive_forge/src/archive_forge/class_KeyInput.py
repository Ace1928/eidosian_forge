from . import interaction
from .input_device import InputDevice
from .interaction import Interaction
from .interaction import Pause
class KeyInput(InputDevice):

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.type = interaction.KEY

    def encode(self) -> dict:
        return {'type': self.type, 'id': self.name, 'actions': [acts.encode() for acts in self.actions]}

    def create_key_down(self, key) -> None:
        self.add_action(TypingInteraction(self, 'keyDown', key))

    def create_key_up(self, key) -> None:
        self.add_action(TypingInteraction(self, 'keyUp', key))

    def create_pause(self, pause_duration: float=0) -> None:
        self.add_action(Pause(self, pause_duration))