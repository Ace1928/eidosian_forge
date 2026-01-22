from __future__ import annotations
import abc
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np
class ColorMapKind(Enum):
    sequential = 'sequential'
    diverging = 'diverging'
    qualitative = 'qualitative'
    cyclic = 'cyclic'
    miscellaneous = 'miscellaneous'