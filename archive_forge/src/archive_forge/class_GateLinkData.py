from abc import ABC
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import numpy as np
from qiskit import circuit
from qiskit.visualization.timeline import types
from qiskit.visualization.exceptions import VisualizationError
class GateLinkData(ElementaryData):
    """A special drawing data type that represents bit link of multi-bit gates.

    Note this object takes multiple bits and dedicates them to the bit link.
    This may appear as a line on the canvas.
    """

    def __init__(self, xval: types.Coordinate, bits: List[types.Bits], styles: Dict[str, Any]=None):
        """Create new bit link.

        Args:
            xval: Horizontal coordinate that the object is drawn.
            bits: Bit associated to this object.
            styles: Style keyword args of the object. This conforms to `matplotlib`.
        """
        super().__init__(data_type=types.LineType.GATE_LINK, xvals=[xval], yvals=[0], bits=bits, meta=None, styles=styles)