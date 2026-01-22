from __future__ import annotations
import re
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import join_keys, match
from ..exceptions import PlotnineError, PlotnineWarning
from .facet import (
from .strips import Strips, strip

    Compute the rows and columns given the number of plots.
    