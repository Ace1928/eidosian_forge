import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
class LoggerHistory(logging.Handler):
    history = []

    def emit(self, message):
        LoggerHistory.history = [message] + LoggerHistory.history[:100]

    @classmethod
    def clear_history(cls):
        del cls.history[:]

    def flush(self):
        super(LoggerHistory, self).flush()
        self.clear_history()