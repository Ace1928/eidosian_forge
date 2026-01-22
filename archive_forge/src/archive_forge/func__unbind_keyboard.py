from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _unbind_keyboard(self):
    keyboard = self._keyboard
    if keyboard:
        keyboard.unbind(on_key_down=self.keyboard_on_key_down, on_key_up=self.keyboard_on_key_up, on_textinput=self.keyboard_on_textinput)
        if self._requested_keyboard:
            keyboard.release()
            self._keyboard = None
            self._requested_keyboard = False
            del FocusBehavior._keyboards[keyboard]
        else:
            FocusBehavior._keyboards[keyboard] = None