from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import order_as_data_mapping, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..positions import position_nudge
from .geom import geom

            Format items in series

            Missing values are preserved as None
            