import copy
from dataclasses import astuple, dataclass
from typing import (
import matplotlib as mpl
import matplotlib.collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1
from cirq.devices import grid_qubit
from cirq.vis import vis_utils
@dataclass
class PolygonUnit:
    """Dataclass to store information about a single polygon unit to plot on the heatmap

    For single (grid) qubit heatmaps, the polygon is a square.
    For two (grid) qubit interaction heatmaps, the polygon is a hexagon.

    Args:
        polygon: Vertices of the polygon to plot.
        value: The value for the heatmap coloring.
        center: The center point of the polygon where annotation text should be printed.
        annot: The annotation string to print on the coupler.

    """
    polygon: Polygon
    value: float
    center: Point
    annot: Optional[str]