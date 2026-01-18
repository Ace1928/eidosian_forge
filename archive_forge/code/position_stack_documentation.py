from __future__ import annotations
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import remove_missing
from ..exceptions import PlotnineWarning
from .position import position

        Stack overlapping intervals.

        Assumes that each set has the same horizontal position
        