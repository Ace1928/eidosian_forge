from __future__ import annotations
import datetime
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.utils import get_cwidth
class SpinningWheel(Formatter):
    """
    Display a spinning wheel.
    """
    characters = '/-\\|'

    def format(self, progress_bar: ProgressBar, progress: ProgressBarCounter[object], width: int) -> AnyFormattedText:
        index = int(time.time() * 3) % len(self.characters)
        return HTML('<spinning-wheel>{0}</spinning-wheel>').format(self.characters[index])

    def get_width(self, progress_bar: ProgressBar) -> AnyDimension:
        return D.exact(1)