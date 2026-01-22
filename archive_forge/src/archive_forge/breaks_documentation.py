from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max

            Calculates the smallest distance in the log scale between the
            currently selected breaks and a new candidate 'x'
            