from __future__ import annotations
import typing
from warnings import warn
import numpy as np
from .._utils import groupby_apply, resolution, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
from .geom_path import geom_path

        Draw a point in the box

        Parameters
        ----------
        data : Series
            Data Row
        da : DrawingArea
            Canvas
        lyr : layer
            Layer

        Returns
        -------
        out : DrawingArea
        