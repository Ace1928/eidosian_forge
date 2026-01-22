import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
def _check_symmetry(self):
    """
        Calculates symmetry

        Returns:
            Bool: True if symmetric, False otherwise
        """
    return self.graph.is_symmetric()