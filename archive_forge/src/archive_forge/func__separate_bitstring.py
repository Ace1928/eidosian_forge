import numpy as np
from qiskit.exceptions import QiskitError
def _separate_bitstring(bitstring, creg_sizes):
    """Separate a bitstring according to the registers defined in the result header."""
    substrings = []
    running_index = 0
    for _, size in reversed(creg_sizes):
        substrings.append(bitstring[running_index:running_index + size])
        running_index += size
    return ' '.join(substrings)