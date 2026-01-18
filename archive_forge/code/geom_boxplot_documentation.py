from __future__ import annotations
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import (
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..positions import position_dodge2
from ..positions.position import position
from .geom import geom
from .geom_crossbar import geom_crossbar
from .geom_point import geom_point
from .geom_segment import geom_segment
Flatten list-likes