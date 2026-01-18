from kivy.clock import Clock
from kivy.config import Config
from kivy.properties import OptionProperty, ObjectProperty, \
from time import time
def trigger_action(self, duration=0.1):
    """Trigger whatever action(s) have been bound to the button by calling
        both the on_press and on_release callbacks.

        This is similar to a quick button press without using any touch events,
        but note that like most kivy code, this is not guaranteed to be safe to
        call from external threads. If needed use
        :class:`Clock <kivy.clock.Clock>` to safely schedule this function and
        the resulting callbacks to be called from the main thread.

        Duration is the length of the press in seconds. Pass 0 if you want
        the action to happen instantly.

        .. versionadded:: 1.8.0
        """
    self._do_press()
    self.dispatch('on_press')

    def trigger_release(dt):
        self._do_release()
        self.dispatch('on_release')
    if not duration:
        trigger_release(0)
    else:
        Clock.schedule_once(trigger_release, duration)