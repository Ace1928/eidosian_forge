from kivy.config import Config
from kivy.logger import Logger
from kivy.input import providers
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def _get_provider_key(self, event):
    """Returns the provider key for the event if the provider is configured
        for calibration.
        """
    input_type = self.provider_map.get(event.__class__)
    key = '({})'.format(input_type)
    if input_type and key in self.devices:
        return key