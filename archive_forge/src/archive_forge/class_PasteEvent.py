import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
class PasteEvent(Event):
    """Multiple keypress events combined, likely from copy/paste.

    The events attribute contains a list of keypress event strings.
    """

    def __init__(self) -> None:
        self.events: List[str] = []

    def __repr__(self) -> str:
        return '<Paste Event with data: %r>' % self.events

    @property
    def name(self) -> str:
        return repr(self)