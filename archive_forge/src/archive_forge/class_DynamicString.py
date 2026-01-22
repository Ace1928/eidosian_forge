from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class DynamicString(str, Enum):
    """The string which is dynamically updated at the time of drawing.

    SCALE: A temporal value of chart scaling factor.
    """
    SCALE = '@scale'