import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _render_bar(self):
    if self.enable_bar and (self._last_task is None or self._last_task.show_bar):
        spin_str = '/-\\|'[self._spin_pos % 4]
        self._spin_pos += 1
        cols = 20
        if self._last_task is None:
            completion_fraction = 0
            self._fraction = 0
        else:
            completion_fraction = self._last_task._overall_completion_fraction() or 0
        if completion_fraction < self._fraction and 'progress' in debug.debug_flags:
            debug.set_trace()
        self._fraction = completion_fraction
        markers = int(round(float(cols) * completion_fraction)) - 1
        bar_str = '[' + ('#' * markers + spin_str).ljust(cols) + '] '
        return bar_str
    elif self._last_task is None or self._last_task.show_spinner:
        spin_str = '/-\\|'[self._spin_pos % 4]
        self._spin_pos += 1
        return spin_str + ' '
    else:
        return ''