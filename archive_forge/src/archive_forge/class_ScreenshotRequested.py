from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@event_class('Overlay.screenshotRequested')
@dataclass
class ScreenshotRequested:
    """
    Fired when user asks to capture screenshot of some area on the page.
    """
    viewport: page.Viewport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScreenshotRequested:
        return cls(viewport=page.Viewport.from_json(json['viewport']))