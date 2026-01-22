from __future__ import annotations
import warnings
from collections import UserString
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
def all_numeric(x):
    for x_i in x:
        if not isinstance(x_i, Number):
            return False
    return True