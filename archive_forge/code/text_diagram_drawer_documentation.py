from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
Horizontally stack text diagrams.

        Args:
            diagrams: The diagrams to stack, ordered from left to right.
            padding_resolver: A function that takes a list of paddings
                specified for a row and returns the padding to use in the
                stacked diagram. Defaults to raising ValueError if the diagrams
                to stack contain inconsistent padding in any row, including
                if some specify a padding and others don't.

        Raises:
            ValueError: Inconsistent padding cannot be resolved.

        Returns:
            The horizontally stacked diagram.
        