from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _set_on_focus_next(self, instance, value):
    """If changing focus, ensure your code does not create an infinite loop.
        eg:
        ```python
        widget.focus_next = widget
        widget.focus_previous = widget
        ```
        """
    next_ = self._old_focus_next
    if next_ is value:
        return
    if isinstance(next_, FocusBehavior):
        next_.focus_previous = None
    self._old_focus_next = value
    if value is None or value is StopIteration:
        return
    if not isinstance(value, FocusBehavior):
        raise ValueError('focus_next accepts only objects based on FocusBehavior, or the `StopIteration` class.')
    value.focus_previous = self