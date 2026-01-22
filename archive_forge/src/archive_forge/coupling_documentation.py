import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
Draws the coupling map.

        This function calls the :func:`~rustworkx.visualization.graphviz_draw` function from the
        ``rustworkx`` package to draw the :class:`CouplingMap` object.

        Returns:
            PIL.Image: Drawn coupling map.

        