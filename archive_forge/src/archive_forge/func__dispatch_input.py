import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def _dispatch_input(self, *ev):
    if ev in self.input_events:
        self.input_events.remove(ev)
    self.input_events.append(ev)