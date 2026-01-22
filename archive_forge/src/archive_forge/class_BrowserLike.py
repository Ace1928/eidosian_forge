from __future__ import annotations
import logging # isort:skip
import webbrowser
from os.path import abspath
from typing import Literal, Protocol, cast
from ..settings import settings
class BrowserLike(Protocol):
    """ Interface for browser-like objects.

    """

    def open(self, url: str, new: TargetCode=..., autoraise: bool=...) -> bool:
        ...