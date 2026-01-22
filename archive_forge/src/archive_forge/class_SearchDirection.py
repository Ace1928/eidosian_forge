from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING
from .application.current import get_app
from .filters import FilterOrBool, is_searching, to_filter
from .key_binding.vi_state import InputMode
class SearchDirection(Enum):
    FORWARD = 'FORWARD'
    BACKWARD = 'BACKWARD'