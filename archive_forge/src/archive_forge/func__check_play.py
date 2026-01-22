from kivy.clock import Clock
from kivy.utils import platform, deprecated
from kivy.core.audio import Sound, SoundLoader
def _check_play(self, dt):
    if self._channel is None:
        return False
    if self._channel.get_busy():
        return
    if self.loop:

        def do_loop(dt):
            self.play()
        Clock.schedule_once(do_loop)
    else:
        self.stop()
    return False