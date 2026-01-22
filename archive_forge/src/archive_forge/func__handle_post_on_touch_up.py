from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
@staticmethod
def _handle_post_on_touch_up(touch):
    """ Called by window after each touch has finished.
        """
    touches = FocusBehavior.ignored_touch
    if touch in touches:
        touches.remove(touch)
        return
    if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown', 'scrollleft', 'scrollright'):
        return
    for focusable in list(FocusBehavior._keyboards.values()):
        if focusable is None or not focusable.unfocus_on_touch:
            continue
        focusable.focus = False