from __future__ import annotations
import typing
from copy import deepcopy
import pandas as pd
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping import aes
from abc import ABC
def to_layer(self) -> layer:
    """
        Make a layer that represents this stat

        Returns
        -------
        out :
            Layer
        """
    from ..geoms.geom import geom
    return layer.from_geom(geom.from_stat(self))