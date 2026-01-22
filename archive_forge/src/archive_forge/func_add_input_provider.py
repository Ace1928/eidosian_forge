import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def add_input_provider(self, provider, auto_remove=False):
    """Add a new input provider to listen for touch events.
        """
    if provider not in self.input_providers:
        self.input_providers.append(provider)
        if auto_remove:
            self.input_providers_autoremove.append(provider)