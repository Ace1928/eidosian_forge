from __future__ import annotations
from enum import Enum
from typing import NamedTuple, Union, Optional, NewType, Any, List
import numpy as np
from qiskit import pulse
class SymbolType(str, Enum):
    """
    Symbol data type.

    FRAME: Assigned to objects that represent symbol of frame.
    SNAPSHOT: Assigned to objects that represent symbol of snapshot.
    """
    FRAME = 'Symbol.Frame'
    SNAPSHOT = 'Symbol.Snapshot'