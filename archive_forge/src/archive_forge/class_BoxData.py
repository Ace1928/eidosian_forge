from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError
class BoxData(ElementaryData):
    """Drawing object that represents box shape.

    This is the counterpart of `matplotlib.patches.Rectangle`.
    """

    def __init__(self, data_type: str | Enum, xvals: np.ndarray | list[types.Coordinate], yvals: np.ndarray | list[types.Coordinate], channels: Channel | list[Channel] | None=None, meta: dict[str, Any] | None=None, ignore_scaling: bool=False, styles: dict[str, Any] | None=None):
        """Create new box.

        Args:
            data_type: String representation of this drawing.
            xvals: Left and right coordinate that the object is drawn.
            yvals: Top and bottom coordinate that the object is drawn.
            channels: Pulse channel object bound to this drawing.
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.

        Raises:
            VisualizationError: When number of data points are not equals to 2.
        """
        if len(xvals) != 2 or len(yvals) != 2:
            raise VisualizationError('Length of data points are not equals to 2.')
        super().__init__(data_type=data_type, xvals=xvals, yvals=yvals, channels=channels, meta=meta, ignore_scaling=ignore_scaling, styles=styles)