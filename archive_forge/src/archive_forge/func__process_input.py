from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def _process_input(self, event):
    for parsed_event in self._parser.feed(event.new):
        self.process_event(parsed_event)