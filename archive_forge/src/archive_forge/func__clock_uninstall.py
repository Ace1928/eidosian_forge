from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def _clock_uninstall(self):
    if self._widgets or not self._clock_installed:
        return
    self._clock_installed = False
    if self._update_ev is not None:
        self._update_ev.cancel()
        self._update_ev = None