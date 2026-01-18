import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
Constructs a Cirq circuit from Quirk's JSON format.

    Args:
        data: Data parsed from quirk's JSON representation.
        qubits: Qubits to use in the circuit. See quirk_url_to_circuit.
        extra_cell_makers: Non-standard Quirk cells to accept. See
            quirk_url_to_circuit.
        quirk_url: If given, the original URL from which the JSON was parsed, as
            described in quirk_url_to_circuit.
        max_operation_count: If the number of operations in the circuit would
            exceed this value, the method raises a `ValueError` instead of
            attempting to construct the circuit. This is important to specify
            for servers parsing unknown input, because Quirk's format allows for
            a billion laughs attack in the form of nested custom gates.

    Examples:

    >>> print(cirq.quirk_json_to_circuit(
    ...     {"cols":[["H"], ["•", "X"]]}
    ... ))
    0: ───H───@───
              │
    1: ───────X───

    Returns:
        The parsed circuit.

    Raises:
        ValueError: Invalid circuit URL, or circuit would be larger than
            `max_operations_count`.
    