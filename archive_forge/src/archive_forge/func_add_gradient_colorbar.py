from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
def add_gradient_colorbar(auxbox: AuxTransformBox, colors: Sequence[str], elements: GuideElementsColorbar, raster: bool=False):
    """
    Add an interpolated gradient colorbar to DrawingArea
    """
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import ListedColormap
    if len(colors) == 1:
        colors = [colors[0], colors[0]]
    nbreak = len(colors)
    if elements.is_vertical:
        colorbar_height = elements.key_height
        colorbar_width = elements.key_width
        mesh_width = 1
        mesh_height = nbreak - 1
        linewidth = colorbar_height / mesh_height
        x = np.array([0, colorbar_width])
        y = np.arange(0, nbreak) * linewidth
        X, Y = np.meshgrid(x, y)
        Z = Y / y.max()
    else:
        colorbar_width = elements.key_height
        colorbar_height = elements.key_width
        mesh_width = nbreak - 1
        mesh_height = 1
        linewidth = colorbar_width / mesh_width
        x = np.arange(0, nbreak) * linewidth
        y = np.array([0, colorbar_height])
        X, Y = np.meshgrid(x, y)
        Z = X / x.max()
    coordinates = np.stack([X, Y], axis=-1)
    cmap = ListedColormap(colors)
    coll = QuadMesh(coordinates, antialiased=False, shading='gouraud', cmap=cmap, array=Z.ravel(), rasterized=raster)
    auxbox.add_artist(coll)