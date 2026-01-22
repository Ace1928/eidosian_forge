from abc import ABC, abstractmethod
from typing import Any
from qiskit.visualization.pulse_v2 import core
Get image data to return.

        Args:
            interactive: When set `True` show the circuit in a new window.
                This depends on the matplotlib backend being used supporting this.

        Returns:
            Image data. This depends on the plotter API.
        