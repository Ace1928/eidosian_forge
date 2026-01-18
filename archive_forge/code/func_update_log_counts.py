from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
def update_log_counts(self, event):
    title = []
    if self._number_of_errors:
        title.append(f'<span style="color:rgb(190,0,0);">errors: </span>{self._number_of_errors}')
    if self._number_of_warnings:
        title.append(f'<span style="color:rgb(190,160,20);">w: </span>{self._number_of_warnings}')
    if self._number_of_infos:
        title.append(f'i: {self._number_of_infos}')
    self.title = ', '.join(title)