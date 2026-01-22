from __future__ import annotations
import logging # isort:skip
import os
from typing import (
class PointGeometry(TypedDict):
    type: Literal['point']
    sx: float
    sy: float