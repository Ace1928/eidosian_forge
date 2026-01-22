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
class Rainbow(Formatter):
    """
    For the fun. Add rainbow colors to any of the other formatters.
    """
    colors = ['#%.2x%.2x%.2x' % _hue_to_rgb(h / 100.0) for h in range(0, 100)]

    def __init__(self, formatter: Formatter) -> None:
        self.formatter = formatter

    def format(self, progress_bar: ProgressBar, progress: ProgressBarCounter[object], width: int) -> AnyFormattedText:
        result = self.formatter.format(progress_bar, progress, width)
        result = explode_text_fragments(to_formatted_text(result))
        result2: StyleAndTextTuples = []
        shift = int(time.time() * 3) % len(self.colors)
        for i, (style, text, *_) in enumerate(result):
            result2.append((style + ' ' + self.colors[(i + shift) % len(self.colors)], text))
        return result2

    def get_width(self, progress_bar: ProgressBar) -> AnyDimension:
        return self.formatter.get_width(progress_bar)