from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _set_on_focus_previous(self, instance, value):
    prev = self._old_focus_previous
    if prev is value:
        return
    if isinstance(prev, FocusBehavior):
        prev.focus_next = None
    self._old_focus_previous = value
    if value is None or value is StopIteration:
        return
    if not isinstance(value, FocusBehavior):
        raise ValueError('focus_previous accepts only objects basedon FocusBehavior, or the `StopIteration` class.')
    value.focus_next = self