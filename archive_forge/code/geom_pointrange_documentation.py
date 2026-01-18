from __future__ import annotations
import typing
from copy import copy
from ..doctools import document
from .geom import geom
from .geom_linerange import geom_linerange
from .geom_path import geom_path
from .geom_point import geom_point

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
        