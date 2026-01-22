from .interaction import Interaction
from .wheel_input import WheelInput
class WheelActions(Interaction):

    def __init__(self, source: WheelInput=None):
        if not source:
            source = WheelInput('wheel')
        super().__init__(source)

    def pause(self, duration: float=0):
        self.source.create_pause(duration)
        return self

    def scroll(self, x=0, y=0, delta_x=0, delta_y=0, duration=0, origin='viewport'):
        self.source.create_scroll(x, y, delta_x, delta_y, duration, origin)
        return self