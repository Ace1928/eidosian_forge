from __future__ import annotations
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, copy_missing_columns, resolution, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
from .geom_polygon import geom_polygon
from .geom_segment import geom_segment
Flatten list-likes