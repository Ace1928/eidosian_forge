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
class TimeElapsed(Formatter):
    """
    Display the elapsed time.
    """

    def format(self, progress_bar: ProgressBar, progress: ProgressBarCounter[object], width: int) -> AnyFormattedText:
        text = _format_timedelta(progress.time_elapsed).rjust(width)
        return HTML('<time-elapsed>{time_elapsed}</time-elapsed>').format(time_elapsed=text)

    def get_width(self, progress_bar: ProgressBar) -> AnyDimension:
        all_values = [len(_format_timedelta(c.time_elapsed)) for c in progress_bar.counters]
        if all_values:
            return max(all_values)
        return 0