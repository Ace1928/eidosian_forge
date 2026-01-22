from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _set_keyboard(self, value):
    focus = self.focus
    keyboard = self._keyboard
    keyboards = FocusBehavior._keyboards
    if keyboard:
        self.focus = False
        if self._keyboard:
            del keyboards[keyboard]
    if value and value not in keyboards:
        keyboards[value] = None
    self._keyboard = value
    self.focus = focus