from typing import List, Optional, Sequence
import cirq
def _initial_time(width, depth, sweeps):
    """Estimates the initiation time of a circuit.

    This estimate includes tasks like electronics setup, gate compiling,
    and throughput of constant-time data.

    This time depends on the size of the circuits being compiled
    (width and depth) and also includes a factor for the number of
    times the compilation is done (sweeps).  Since sweeps save some of
    the work, this factor scales less than one.

    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
    """
    return (width / 8 * (depth / 125) + width / 12) * max(1, sweeps / 5)