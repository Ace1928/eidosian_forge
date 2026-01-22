from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _bind_keyboard(self):
    self._ensure_keyboard()
    keyboard = self._keyboard
    if not keyboard or self.disabled or (not self.is_focusable):
        self.focus = False
        return
    keyboards = FocusBehavior._keyboards
    old_focus = keyboards[keyboard]
    if old_focus:
        old_focus.focus = False
    keyboards[keyboard] = self
    keyboard.bind(on_key_down=self.keyboard_on_key_down, on_key_up=self.keyboard_on_key_up, on_textinput=self.keyboard_on_textinput)