from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError
Create new box.

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
        