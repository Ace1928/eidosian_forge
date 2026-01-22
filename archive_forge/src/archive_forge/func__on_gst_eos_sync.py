from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath
def _on_gst_eos_sync(self):
    Clock.schedule_once(self._on_gst_eos, 0)