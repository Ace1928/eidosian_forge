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
def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
    """
    Take hue between 0 and 1, return (r, g, b).
    """
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    q = int(255 * (1.0 - f))
    t = int(255 * (1.0 - (1.0 - f)))
    i %= 6
    return [(255, t, 0), (q, 255, 0), (0, 255, t), (0, q, 255), (t, 0, 255), (255, 0, q)][i]