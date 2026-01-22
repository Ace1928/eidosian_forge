from kivy.clock import Clock
from kivy.logger import Logger
from kivy.core.audio import Sound, SoundLoader
from kivy.weakmethod import WeakMethod
import time
def _do_eos(self, *args):
    if not self.loop:
        self.stop()
    else:
        self.seek(0.0)