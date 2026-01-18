from __future__ import annotations
import typing
import numpy as np
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from .geom import geom
from .geom_path import geom_path

        Draw a rectangle in the box

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
        