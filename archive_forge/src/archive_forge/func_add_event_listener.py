import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def add_event_listener(self, listener):
    """Add a new event listener for getting touch events.
        """
    if listener not in self.event_listeners:
        self.event_listeners.append(listener)