from __future__ import annotations
import warnings
from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from enum import Enum
import numpy as np
from qiskit import circuit
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import drawings, types
from qiskit.visualization.timeline.stylesheet import QiskitTimelineStyle
def y_coords(link: drawings.GateLinkData):
    return np.array([self.assigned_coordinates.get(bit, np.nan) for bit in link.bits])