import sys
from typing import TYPE_CHECKING, Optional, Union
from .jupyter import JupyterMixin
from .segment import Segment
from .style import Style
from ._emoji_codes import EMOJI
from ._emoji_replace import _emoji_replace
class NoEmoji(Exception):
    """No emoji by that name."""