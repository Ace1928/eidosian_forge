from bokeh.core.properties import (
from bokeh.events import ModelEvent
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class KeystrokeEvent(ModelEvent):
    event_name = 'keystroke'

    def __init__(self, model, key=None):
        self.key = key
        super().__init__(model=model)