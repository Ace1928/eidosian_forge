from __future__ import annotations
import typing
from typing import Sized
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import order_as_data_mapping
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..mapping import aes
from .geom import geom
from .geom_path import geom_path
from .geom_segment import geom_segment

        Plot all groups
        