from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp

            Utility method that determines how to build the list of single-qubit Pauli X operators and
            the list of corresponding qubit indices from the stacked symmetries.
            This method is successively applied to Z type, X type and Y type symmetries (in this order)
            to build the letter at position (col) of the Pauli word corresponding to the symmetry at
            position (row).

            Args:
                row (int): Index of the symmetry for which the single-qubit Pauli X operator is being
                    built.
                col (int): Index of the letter in the Pauli word corresponding to the single-qubit Pauli
                    X operator.
                idx_test (list): List of possibilities for the stacked symmetries at all other rows
                    than row.
                row_test (list): List of possibilities for the stacked symmetries at row.

            Returns:
                Whether or not this symmetry type should be used to build this letter of this
                single-qubit Pauli X operator.
            