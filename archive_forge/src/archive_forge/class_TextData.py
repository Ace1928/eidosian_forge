from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError
class TextData(ElementaryData):
    """Drawing object to represent object appears as a text.

    This is the counterpart of `matplotlib.pyplot.text`.
    """

    def __init__(self, data_type: str | Enum, xvals: np.ndarray | list[types.Coordinate], yvals: np.ndarray | list[types.Coordinate], text: str, latex: str | None=None, channels: Channel | list[Channel] | None=None, meta: dict[str, Any] | None=None, ignore_scaling: bool=False, styles: dict[str, Any] | None=None):
        """Create new drawing.

        Args:
            data_type: String representation of this drawing.
            channels: Pulse channel object bound to this drawing.
            xvals: Series of horizontal coordinate that the object is drawn.
            yvals: Series of vertical coordinate that the object is drawn.
            text: String to show in the canvas.
            latex: Latex representation of the text (if backend supports latex drawing).
            meta: Meta data dictionary of the object.
            ignore_scaling: Set ``True`` to disable scaling.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        self.text = text
        self.latex = latex or ''
        super().__init__(data_type=data_type, xvals=xvals, yvals=yvals, channels=channels, meta=meta, ignore_scaling=ignore_scaling, styles=styles)